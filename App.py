import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import time
import sys
sys.path.insert(0, '.')
from src.generate_data import generate_session

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Welding Coach",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/error_detector.pkl", "rb") as f:
        clf, feature_cols = pickle.load(f)
    return clf, feature_cols

clf, FEATURE_COLS = load_model()

# ── Constants ─────────────────────────────────────────────────────────────────
IDEAL = {
    "work_angle":   (85,  95),
    "travel_speed": (5,   12),
    "arc_length":   (4,   8),
    "tremor":       (0,   0.06),
    "voltage":      (20,  24),
}
HARD = {
    "work_angle":   (70,  110),
    "travel_speed": (2,   15),
    "arc_length":   (3,   10),
    "tremor":       (0,   0.10),
    "voltage":      (19,  25),
}
LABELS = {
    "work_angle":   "Work angle (°)",
    "travel_speed": "Travel speed (mm/s)",
    "arc_length":   "Arc length (mm)",
    "tremor":       "Torch tremor",
    "voltage":      "Voltage (V)",
}

ALL_RULES = [
    ("arc_length",   lambda v: v > 10,                          "Move torch closer — arc length too long",   "error"),
    ("arc_length",   lambda v: v < 3,                           "Too close — risk of stubbing electrode",    "error"),
    ("work_angle",   lambda v: v < 70 or v > 110,               "Fix work angle — aim for 90° to workpiece","error"),
    ("travel_speed", lambda v: v > 15,                          "Slow down — travel speed too fast",         "error"),
    ("travel_speed", lambda v: v < 2,                           "Speed up — travel rate too slow",           "error"),
    ("tremor",       lambda v: v > 0.10,                        "Steady hand — reduce torch shake",          "error"),
    ("voltage",      lambda v: v < 19 or v > 25,                "Check voltage — outside WPS range",         "error"),
    ("arc_length",   lambda v: 8 < v <= 10,                     "Arc slightly long — move a little closer",  "warning"),
    ("arc_length",   lambda v: 3 <= v < 4,                      "Arc slightly short — ease back a little",   "warning"),
    ("work_angle",   lambda v: (70 <= v < 85) or (95 < v <= 110),"Angle drifting — re-centre to 90°",       "warning"),
    ("travel_speed", lambda v: 12 < v <= 15,                    "Speed slightly fast — ease off",            "warning"),
    ("travel_speed", lambda v: 2 <= v < 5,                      "Speed slightly slow — pick up pace",        "warning"),
    ("tremor",       lambda v: 0.06 < v <= 0.10,                "Hand not fully steady — relax your grip",   "warning"),
    ("voltage",      lambda v: (19 <= v < 20) or (24 < v <= 25),"Voltage slightly off — check WPS",         "warning"),
]

WINDOW = 50

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(window_df):
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

def get_breaches(row):
    vals = {k: getattr(row, k) for k in IDEAL}
    seen = set()
    breaches = []
    for feat, cond, msg, sev in ALL_RULES:
        key = feat + sev
        if key not in seen and cond(vals[feat]):
            if sev == "warning" and (feat + "error") in seen:
                continue
            breaches.append({"feat": feat, "msg": msg, "sev": sev,
                              "label": LABELS[feat], "val": vals[feat]})
            seen.add(key)
    return breaches

def get_metric_status(key, val, breaches):
    breach_feats = {b["feat"]: b["sev"] for b in breaches}
    if key in breach_feats:
        return breach_feats[key]
    lo, hi = IDEAL[key]
    return "good" if lo <= val <= hi else "neutral"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 AI Welding Coach")
    st.caption("Real-time adaptive guidance engine")
    st.divider()

    skill = st.radio("Session type", ["Beginner", "Expert"],
                     horizontal=True)
    duration = st.slider("Session duration (s)", 10, 60, 20)
    speed = st.select_slider("Playback speed",
                             options=["0.5x", "1x", "2x", "5x", "10x"],
                             value="2x")
    SPEED_MAP = {"0.5x": 0.5, "1x": 1, "2x": 2, "5x": 5, "10x": 10}
    playback_speed = SPEED_MAP[speed]

    generate = st.button("Generate session", type="primary", use_container_width=True)
    run      = st.button("▶ Run coaching session", use_container_width=True)

    st.divider()
    st.caption("**Feature importance (model)**")
    importance_data = {
        "Work angle":   0.227,
        "Arc length":   0.215,
        "Travel speed": 0.175,
        "Travel angle": 0.093,
        "Speed std":    0.068,
        "Tremor":       0.040,
        "Voltage":      0.030,
    }
    for feat, imp in importance_data.items():
        st.progress(imp, text=f"{feat}: {imp:.1%}")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_df" not in st.session_state:
    st.session_state.session_df = None
if "skill" not in st.session_state:
    st.session_state.skill = "Beginner"

if generate:
    with st.spinner("Generating fresh session..."):
        df = generate_session(skill=skill.lower(), duration_s=duration, hz=100)
        st.session_state.session_df = df
        st.session_state.skill = skill
    st.success(f"{skill} session generated — {len(df):,} frames at 100Hz")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("AI Welding Coach")
st.caption("Real-time adaptive guidance engine — Skillveri simulator")

if st.session_state.session_df is None:
    st.info("Click **Generate session** in the sidebar to begin.")
    st.stop()

df = st.session_state.session_df
current_skill = st.session_state.skill

# ── Metric cards (header row) ─────────────────────────────────────────────────
STATUS_COLORS = {
    "error":   "#e24b4a",
    "warning": "#ef9f27",
    "good":    "#1d9e75",
    "neutral": "#888780",
}

def metric_card(label, value, unit, status, ideal_range):
    color = STATUS_COLORS[status]
    ideal_str = f"{ideal_range[0]}–{ideal_range[1]}"
    st.markdown(f"""
    <div style="background:#1a1d2e;border:1.5px solid {color};border-radius:10px;
                padding:14px 16px;margin-bottom:4px">
      <div style="font-size:11px;color:#666;margin-bottom:4px">{label}
        <span style="color:#444;font-size:10px">ideal {ideal_str}</span></div>
      <div style="font-size:24px;font-weight:600;color:{color}">{value}
        <span style="font-size:12px;color:#555">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

# Placeholders for live update
metric_cols  = st.columns(5)
coach_ph     = st.empty()
score_ph     = st.empty()
chart_ph     = st.empty()
timeline_ph  = st.empty()

# ── Run session ───────────────────────────────────────────────────────────────
if run and st.session_state.session_df is not None:
    df = st.session_state.session_df
    hz = 100
    step = 5   # advance 5 frames per tick

    history = {k: [] for k in ["t","work_angle","travel_speed",
                                "arc_length","tremor","error_prob"]}
    good_frames = 0
    total_frames = 0
    last_coach_t = -99
    last_primary = ""
    COOLDOWN = 2.5

    delay = max(0.01, step / hz / playback_speed)

    for i in range(WINDOW, len(df), step):
        window_df = df.iloc[i-WINDOW:i]
        row       = df.iloc[i]
        t         = row.t

        # Model inference
        feats = extract_features(window_df)
        X     = np.array([[feats[c] for c in FEATURE_COLS]])
        is_err= int(clf.predict(X)[0])
        prob  = float(clf.predict_proba(X)[0][1])

        total_frames += 1
        if not is_err:
            good_frames += 1

        score = round(good_frames / total_frames * 100)

        # Breaches
        breaches = get_breaches(row)
        primary  = breaches[0] if breaches else None

        # ── Metric cards ──────────────────────────────────────────────────────
        keys   = ["work_angle","travel_speed","arc_length","tremor","voltage"]
        vals   = [row.work_angle, row.travel_speed, row.arc_length,
                  row.tremor, row.voltage]
        units  = ["°","mm/s","mm","","V"]
        ideals = [IDEAL[k] for k in keys]

        for col, key, val, unit, ideal in zip(
                metric_cols, keys, vals, units, ideals):
            status = get_metric_status(key, val, breaches)
            with col:
                metric_card(LABELS[key],
                            f"{val:.3f}" if key=="tremor" else f"{val:.1f}",
                            unit, status, ideal)

        # ── Coach panel ───────────────────────────────────────────────────────
        with coach_ph.container():
            if not breaches:
                st.success("✓ Good technique — keep it up")
            else:
                has_err = any(b["sev"]=="error" for b in breaches)
                icon    = "🚨" if has_err else "⚠️"
                badge   = f"{sum(1 for b in breaches if b['sev']=='error')} error(s)" \
                          if has_err else f"{len(breaches)} warning(s)"

                with st.container():
                    cols = st.columns([0.05,0.7,0.25])
                    cols[0].markdown(icon)
                    if has_err:
                        cols[1].error(f"**{primary['msg']}**")
                    else:
                        cols[1].warning(f"**{primary['msg']}**")
                    cols[2].markdown(
                        f"<div style='text-align:right;padding-top:8px;"
                        f"color:#888;font-size:12px'>{badge}</div>",
                        unsafe_allow_html=True)

                    if len(breaches) > 1:
                        for rank, b in enumerate(breaches[1:], 2):
                            color = "#f09595" if b["sev"]=="error" else "#ef9f27"
                            st.markdown(
                                f"<div style='padding:5px 12px;margin:2px 0;"
                                f"background:#1a1a1a;border-radius:6px;"
                                f"font-size:12px;color:{color}'>"
                                f"<b>{rank}. {b['label']}</b> — {b['msg']}</div>",
                                unsafe_allow_html=True)

        # ── Score ─────────────────────────────────────────────────────────────
        with score_ph.container():
            score_color = "#1d9e75" if score>=80 else "#ef9f27" if score>=60 else "#e24b4a"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Session score", f"{score}%")
            c2.metric("Good frames",   f"{good_frames}/{total_frames}")
            c3.metric("Active issues", len(breaches))
            c4.metric("Error prob",    f"{prob:.0%}")

        # ── Charts ────────────────────────────────────────────────────────────
        history["t"].append(t)
        history["work_angle"].append(row.work_angle)
        history["travel_speed"].append(row.travel_speed)
        history["arc_length"].append(row.arc_length)
        history["tremor"].append(row.tremor)
        history["error_prob"].append(prob)

        # Keep last 120 points
        for k in history:
            if len(history[k]) > 120:
                history[k] = history[k][-120:]

        with chart_ph.container():
            col1, col2 = st.columns(2)

            def make_fig(key, title, ideal_lo, ideal_hi, color):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history["t"], y=history[key],
                    mode="lines", name="Live",
                    line=dict(color=color, width=1.5)))
                fig.add_hline(y=ideal_lo, line_dash="dash",
                              line_color="#1d9e75", line_width=1)
                fig.add_hline(y=ideal_hi, line_dash="dash",
                              line_color="#1d9e75", line_width=1)
                fig.update_layout(
                    title=dict(text=title, font=dict(size=12)),
                    height=180, margin=dict(l=0,r=0,t=30,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#1a1d2e",
                    font=dict(color="#888", size=10),
                    showlegend=False,
                    xaxis=dict(showgrid=False, title=""),
                    yaxis=dict(gridcolor="#252840"),
                )
                return fig

            with col1:
                st.plotly_chart(make_fig("work_angle","Work angle (°)",
                                         85,95,"#378add"),
                                use_container_width=True, key=f"wa_{i}")
                st.plotly_chart(make_fig("arc_length","Arc length (mm)",
                                         4,8,"#a78bfa"),
                                use_container_width=True, key=f"al_{i}")
            with col2:
                st.plotly_chart(make_fig("travel_speed","Travel speed (mm/s)",
                                         5,12,"#ef9f27"),
                                use_container_width=True, key=f"ts_{i}")
                st.plotly_chart(make_fig("error_prob","Error probability",
                                         0,0.2,"#e24b4a"),
                                use_container_width=True, key=f"ep_{i}")

        timeline_ph.caption(
            f"t = {t:.2f}s  |  good: {good_frames}/{total_frames} frames  "
            f"|  active issues: {len(breaches)}  |  skill: {current_skill}")

        time.sleep(delay)

    st.balloons()
    st.success(f"Session complete — final score: {score}%")