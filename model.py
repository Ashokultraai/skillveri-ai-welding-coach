"""
model.py - AI Welding Coach
============================
Trains the Random Forest error detector.
Run this AFTER generate_data.py.

Steps:
  1. Load welding_data.csv
  2. Extract 14 features from 0.5s sliding windows
  3. Train / test split (80/20)
  4. Train Random Forest (100 trees)
  5. Evaluate and print results
  6. Save model to models/error_detector.pkl
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# STEP 1 - Load data
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 - Loading raw sensor data")
print("=" * 60)

data_path = os.path.join("data", "welding_data.csv")
df = pd.read_csv(data_path)

print(f"Total frames    : {len(df):,}")
print(f"Expert frames   : {len(df[df.skill == 'expert']):,}")
print(f"Beginner frames : {len(df[df.skill == 'beginner']):,}")
print(f"Error frames    : {df.error.sum():,} ({df.error.mean()*100:.1f}%)\n")

# -----------------------------------------------------------------------------
# STEP 2 - Feature extraction
# -----------------------------------------------------------------------------
"""
Why sliding windows?
A single frame (1/100s) is too noisy. A 0.5s window (50 frames)
captures the TREND - is the welder consistently making an error?

14 features extracted per window:
  Pose:       work_angle, travel_angle (mean + std)
  Dynamic:    travel_speed, arc_length (mean + std), tremor, jerk
  Electrical: voltage_std, current_std, wfs_std
"""
print("=" * 60)
print("STEP 2 - Feature extraction (50-frame sliding windows)")
print("=" * 60)

WINDOW = 50  # 50 frames = 0.5 seconds at 100Hz

def extract_features(df, window=WINDOW):
    rows = []
    for i in range(window, len(df)):
        w = df.iloc[i - window:i]
        row = {
            "work_angle_mean":    w.work_angle.mean(),
            "work_angle_std":     w.work_angle.std(),
            "travel_angle_mean":  w.travel_angle.mean(),
            "travel_angle_std":   w.travel_angle.std(),
            "travel_speed_mean":  w.travel_speed.mean(),
            "travel_speed_std":   w.travel_speed.std(),
            "arc_length_mean":    w.arc_length.mean(),
            "arc_length_std":     w.arc_length.std(),
            "tremor_mean":        w.tremor.mean(),
            "tremor_max":         w.tremor.max(),
            "jerk":               np.gradient(w.accel_x).std(),
            "voltage_std":        w.voltage.std(),
            "current_std":        w.current.std(),
            "wfs_std":            w.wfs.std(),
            "error":              int(w.error.mean() > 0.3),
        }
        rows.append(row)
    return pd.DataFrame(rows)

features = extract_features(df)
FEATURE_COLS = [c for c in features.columns if c != "error"]

print(f"Feature windows : {len(features):,}")
print(f"Features per win: {len(FEATURE_COLS)}")
print(f"Error windows   : {features.error.sum():,} ({features.error.mean()*100:.1f}%)")
print(f"\nAll 14 features:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2}. {col}")
print()

# -----------------------------------------------------------------------------
# STEP 3 - Train / test split
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 3 - Train / test split (80% / 20%)")
print("=" * 60)

X = features[FEATURE_COLS].values
y = features["error"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}\n")

# -----------------------------------------------------------------------------
# STEP 4 - Train Random Forest
# -----------------------------------------------------------------------------
"""
Random Forest = 100 Decision Trees voting together.

Each tree asks yes/no questions about the features:
  "Is work_angle_mean > 100?" -> yes -> likely error
  "Is arc_length_std > 2.0?"  -> yes -> likely error

Key hyperparameters:
  n_estimators = 100  -> 100 trees (more = more stable)
  max_depth    = 8    -> max 8 questions per tree (prevents overfitting)
  min_samples  = 5    -> leaf needs 5+ samples (prevents overspecific rules)
  max_features = sqrt -> each split uses sqrt(14) ~ 4 features randomly
"""
print("=" * 60)
print("STEP 4 - Training Random Forest (100 trees)")
print("=" * 60)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
print("Training complete.\n")

# -----------------------------------------------------------------------------
# STEP 5 - Evaluate
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 5 - Evaluation on unseen test data")
print("=" * 60)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Good", "Error"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(f"                  Predicted Good   Predicted Error")
print(f"  Actual Good          {cm[0,0]:>6}            {cm[0,1]:>6}")
print(f"  Actual Error         {cm[1,0]:>6}            {cm[1,1]:>6}\n")

# -----------------------------------------------------------------------------
# STEP 6 - Feature importance
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 6 - Feature importance")
print("=" * 60)

importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

for rank, (feat, imp) in enumerate(importances.items(), 1):
    bar = "#" * int(imp * 100)
    print(f"{rank:2}. {feat:<25} {imp:>6.1%}  {bar}")

# -----------------------------------------------------------------------------
# STEP 7 - Save model
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 - Saving model")
print("=" * 60)

models_dir = os.path.join("models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "error_detector.pkl")

with open(model_path, "wb") as f:
    pickle.dump((clf, FEATURE_COLS), f)

print(f"Saved to: {model_path}")
print(f"\nTo load in app.py:")
print(f"  with open('models/error_detector.pkl', 'rb') as f:")
print(f"      clf, FEATURE_COLS = pickle.load(f)")
print(f"\nDone. Run: streamlit run app.py")