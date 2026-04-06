# AI Welding Coach — Real-Time Adaptive Guidance Engine

An AI-powered welding coach built for the Skillveri simulator platform. Detects technique errors in real time from 6DOF sensor data and delivers prioritised coaching feedback through voice, haptic, and visual overlays.

---

## Demo

Open `index.html` directly in any browser — no server or Python needed.

- Click **Expert session** or **Beginner session** to generate a fresh random session
- Press **Play** to start the live coaching simulation
- Watch metrics update in real time with colour-coded severity
- Coaching panel shows all active issues ranked by priority

---

## Project Structure

```
skillveri-ai-welding-coach/
├── generate_data.py     # Step 1 — Synthetic 6DOF sensor data generator
├── model.py             # Step 2 — Feature extraction + Random Forest training
├── coach.py             # Step 3 — Real-time coaching engine (terminal demo)
├── index.html           # Step 4 — Live visual demo (open in browser)
└── Requirements.txt     # Python dependencies
```

> **Note:** `data/` and `models/` folders are auto-generated when you run the scripts. They are not included in the repo.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r Requirements.txt
```

### 2. Generate training data
```bash
python generate_data.py
```
Generates 30,000 frames of synthetic 6DOF torch motion at 100Hz across 5 expert and 5 beginner sessions. Automatically creates `data/welding_data.csv`.

### 3. Train the model
```bash
python model.py
```
Extracts 14 features from 0.5-second sliding windows and trains a Random Forest classifier. Prints step-by-step output including feature importance and confusion matrix. Saves trained model to `models/error_detector.pkl`.

### 4. Run terminal coaching demo (optional)
```bash
python coach.py
```
Runs expert and beginner sessions and prints real-time coaching corrections to the terminal with error confidence scores.

### 5. Open the visual demo
Open `index.html` in Chrome or Edge. No server needed.

---

## System Architecture

```
Input layer
  Tracked welding gun (6DOF) ──┐
  Sim engine (V, A, WFS)  ──── Sensor fusion → Data sync bus (100Hz)
  Public datasets (training)         ↓
                              Feature extraction
                         ┌─────────┼──────────┐
                         ↓         ↓          ↓
                  Technique    Error      Skill
                  classifier  detector  estimator
                         └─────────┼──────────┘
                                   ↓
                        Coaching decision engine
                     (priority queue — one fix at a time)
                    ┌──────────┬──────────┐
                    ↓          ↓          ↓
                 Voice cue  AR overlay  Haptic pulse
                    └──────────┴──────────┘
                                ↓
                      Session store + retrain loop
```

---

## Features Extracted (14 total)

| Group | Feature | Source |
|---|---|---|
| Pose | work_angle_mean, work_angle_std | 6DOF tracking |
| Pose | travel_angle_mean, travel_angle_std | 6DOF tracking |
| Dynamic | travel_speed_mean, travel_speed_std | IMU accelerometer |
| Dynamic | arc_length_mean, arc_length_std | IMU + position |
| Dynamic | tremor_mean, tremor_max | Raw IMU (100Hz) |
| Dynamic | jerk | IMU accel derivative |
| Electrical | voltage_std | Sim engine |
| Electrical | current_std | Sim engine |
| Electrical | wfs_std | Sim engine |

---

## Coaching Priority Order

Errors are ranked by physical severity — the most dangerous defect cause is always addressed first:

| Priority | Metric | Defect caused if wrong |
|---|---|---|
| 1 | Arc length | Porosity, stubbing risk |
| 2 | Work angle | Lack of fusion |
| 3 | Travel speed | Undercut or burn-through |
| 4 | Torch tremor | Inconsistent bead width |
| 5 | Voltage | Arc instability |

---

## Three-Tier Severity System

| Card colour | Border | Meaning |
|---|---|---|
| Green | Normal | Within ideal range |
| Orange | Normal or thick | Outside ideal but within hard limit (warning) |
| Red | Thick | Outside hard limit — defect likely (error) |

The coaching panel always stays consistent with the metric cards — if a card is red, there is always a corresponding line in the coach panel.

---

## Model Performance

```
              Precision   Recall   F1-score
Good              1.00     1.00      1.00
Error             0.99     0.99      0.99
Accuracy                             1.00
```

Trained on 30,000 synthetic frames, 80/20 train/test split, stratified by error rate.

---

## Why Random Forest (not a neural network)?

- Runs in **microseconds** per prediction — needed for 100Hz real-time inference
- **Fully explainable** — feature importance shows exactly what the model learned
- Works on **small datasets** — no GPU, no cloud, runs on Skillveri simulator hardware
- **Easy to retrain** weekly as new session data accumulates
- Work angle and arc length ranked #1 and #2 in feature importance — independently validating our metric selection against AWS WPS standards

---

## Metrics Validation

The 5 coaching metrics were chosen from two converging sources:

1. **AWS WPS standards** — Welding Procedure Specification defines acceptable parameter ranges for certified MIG welding on mild steel
2. **Model feature importance** — the Random Forest independently ranked the same metrics as most predictive of errors

Both sources agreed: work angle and arc length matter most.

---

## Production Roadmap

| What | How |
|---|---|
| Replace synthetic data | Instrument Skillveri sessions from day one — log all 6DOF data |
| Improve model | LSTM on raw IMU sequences once real labelled data is available |
| Add LLM coaching | Context-aware messages ("your angle has drifted for 3 seconds — try anchoring your elbow") |
| Connect to hardware | Skillveri HMD (AR overlay), helmet speaker (TTS), haptic controller |
| Weekly retrain | Session logs → label → fine-tune → deploy |

---

