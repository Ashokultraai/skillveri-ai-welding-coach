"""
coach.py - AI Welding Coach
============================
Terminal demo of the coaching engine.
Run AFTER model.py has created error_detector.pkl.

Usage:
  python src/coach.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.generate_data import generate_session

# Load model using relative path (works on Windows, Mac, Linux)
model_path = os.path.join("models", "error_detector.pkl")
with open(model_path, "rb") as f:
    clf, FEATURE_COLS = pickle.load(f)

# Expert baseline acceptable ranges
EXPERT_BASELINE = {
    "work_angle":   (85.0, 95.0),
    "travel_angle": (10.0, 20.0),
    "travel_speed": (5.0,  12.0),
    "arc_length":   (4.0,   8.0),
    "tremor":       (0.0,   0.06),
}

# Coaching rules in priority order (most dangerous first)
COACHING_RULES = [
    ("arc_length",   lambda v: v > 10,           "Move torch closer — arc length too long",     1),
    ("arc_length",   lambda v: v < 3,            "Too close — risk of stubbing electrode",       1),
    ("work_angle",   lambda v: v < 70 or v > 110,"Adjust work angle — aim for 90°",             1),
    ("travel_speed", lambda v: v > 15,           "Slow down — travel speed too fast",            1),
    ("travel_speed", lambda v: v < 2,            "Speed up — travel rate too slow",              1),
    ("tremor",       lambda v: v > 0.10,         "Steady your hand — reduce torch shake",        2),
    ("voltage",      lambda v: v < 19 or v > 25, "Check voltage — outside WPS range",           2),
]

def extract_window_features(window_df):
    w = window_df
    return {
        "work_angle_mean":   w.work_angle.mean(),
        "work_angle_std":    w.work_angle.std(),
        "travel_angle_mean": w.travel_angle.mean(),
        "travel_angle_std":  w.travel_angle.std(),
        "travel_speed_mean": w.travel_speed.mean(),
        "travel_speed_std":  w.travel_speed.std(),
        "arc_length_mean":   w.arc_length.mean(),
        "arc_length_std":    w.arc_length.std(),
        "tremor_mean":       w.tremor.mean(),
        "tremor_max":        w.tremor.max(),
        "jerk":              float(np.gradient(w.accel_x).std()),
        "voltage_std":       w.voltage.std(),
        "current_std":       w.current.std(),
        "wfs_std":           w.wfs.std(),
    }

def get_coaching_message(window_df):
    feats = extract_window_features(window_df)
    current_vals = {
        "work_angle":   feats["work_angle_mean"],
        "travel_angle": feats["travel_angle_mean"],
        "travel_speed": feats["travel_speed_mean"],
        "arc_length":   feats["arc_length_mean"],
        "tremor":       feats["tremor_mean"],
        "voltage":      22.0,
    }
    for feature, condition, message, severity in COACHING_RULES:
        if condition(current_vals.get(feature, 0)):
            return message, feature, current_vals.get(feature, 0), severity
    return None, None, None, None

def run_demo(skill="beginner", duration_s=15, hz=100, window=50):
    print(f"\n{'='*60}")
    print(f"  AI WELDING COACH - LIVE SESSION ({skill.upper()})")
    print(f"{'='*60}")
    print(f"  Monitoring {duration_s}s of welding at {hz}Hz...\n")

    df = generate_session(skill=skill, duration_s=duration_s, hz=hz)

    last_coached_at = -5
    corrections = []
    error_frames = 0
    total_frames = 0

    for i in range(window, len(df), 10):
        window_df = df.iloc[i-window:i]
        t = df.iloc[i].t
        total_frames += 1

        feat_row = extract_window_features(window_df)
        X = np.array([[feat_row[c] for c in FEATURE_COLS]])
        is_error = clf.predict(X)[0]
        prob = clf.predict_proba(X)[0][1]

        if is_error:
            error_frames += 1

        if is_error and (t - last_coached_at) >= 3.0:
            message, feature, value, severity = get_coaching_message(window_df)
            if message:
                tag = "[URGENT]" if severity == 1 else "[TIP]   "
                print(f"  t={t:5.1f}s  {tag}  {message}")
                print(f"           -> {feature}: {value:.1f}  (confidence: {prob:.0%})")
                corrections.append(message)
                last_coached_at = t

    accuracy = 1 - (error_frames / total_frames)
    print(f"\n{'-'*60}")
    print(f"  SESSION COMPLETE")
    print(f"  Technique score  : {accuracy*100:.0f}%")
    print(f"  Corrections given: {len(corrections)}")
    print(f"  Skill assessed   : {'Advanced' if accuracy > 0.9 else 'Intermediate' if accuracy > 0.7 else 'Beginner'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_demo(skill="expert",   duration_s=15)
    run_demo(skill="beginner", duration_s=15)