import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import time

# Simulation logic representing acoustic/vibrational sensors in RAAC panels
from simulation.generate_signal import generate_signal
from simulation.feature_extraction import extract_features

# Page Configuration
st.set_page_config(page_title="RAAC Structural Analyzer", layout="wide")

# --- UI Styling ---
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main { padding: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏗️ RAAC Structural Integrity Analyzer")
st.write("""
    This system analyzes signal patterns from sensors embedded in **Reinforced Autoclaved Aerated Concrete (RAAC)**. 
    The AI model identifies deviations in acoustic signatures that may indicate structural compromise or material fatigue.
""")

# --- Assets ---
@st.cache_resource
def load_assets():
    # Model path matches the provided project structure
    interpreter = tf.lite.Interpreter(model_path="model/raac_model.tflite")
    interpreter.allocate_tensors()
    scaler = joblib.load("model/scaler.pkl")
    return interpreter, scaler

try:
    interpreter, scaler = load_assets()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Configuration Error: {e}. Please ensure the model and scaler files exist.")
    st.stop()

# --- Dashboard Layout ---
st.sidebar.header("🔍 Inspection Parameters")
inspection_mode = st.sidebar.radio("Inspection Target:", ["Stable Panel", "Degraded Panel (Abnormal)", "Random Scan"])
run_speed = st.sidebar.select_slider("Scan Frequency", options=["Slow", "Moderate", "High-Speed"], value="Moderate")
auto_scan = st.sidebar.toggle("Enable Continuous Monitoring")

delay_map = {"Slow": 2.5, "Moderate": 1.0, "High-Speed": 0.3}

col1, col2 = st.columns([2, 1])

# --- Logic Execution ---
if st.button("Initiate Single Scan") or auto_scan:
    # Set signal logic[cite: 1]
    if inspection_mode == "Random Scan":
        target = np.random.choice([0, 1])
    else:
        target = 1 if "Degraded" in inspection_mode else 0

    # Data Processing[cite: 1]
    signal_data = generate_signal(target)
    metrics = extract_features(signal_data)
    
    # Machine Learning Inference[cite: 1]
    scaled_input = scaler.transform([metrics])
    interpreter.set_tensor(input_details[0]['index'], scaled_input.astype('float32'))
    interpreter.invoke()
    risk_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
    is_compromised = risk_score > 0.5

    with col1:
        st.subheader("Sensor Waveform Analysis")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        line_color = '#d62728' if is_compromised else '#2ca02c'
        ax.plot(signal_data, color=line_color, alpha=0.8, linewidth=2)
        ax.fill_between(range(len(signal_data)), signal_data, color=line_color, alpha=0.1)
        ax.set_title("Acoustic Emission / Vibrational Signature")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col2:
        st.subheader("Diagnostic Outcome")
        
        # Risk Gauge
        if is_compromised:
            st.error(f"### ALERT: STRUCTURAL COMPROMISE\nConfidence Score: {risk_score:.2%}")
            st.warning("Recommendation: Immediate onsite physical inspection required.")
        else:
            st.success(f"### STATUS: STABLE\nConfidence Score: {(1 - risk_score):.2%}")
            st.info("Recommendation: Schedule routine scan in 6 months.")

        # Data Table
        st.write("**Panel Telemetry:**")
        df_metrics = pd.DataFrame({
            "Metric": ["Average Signal", "Signal Variance", "Minimum Peak", "Maximum Peak", "CV Ratio"],
            "Value": [f"{m:.4f}" for m in metrics]
        })
        st.table(df_metrics.set_index("Metric"))

    # Technical Overview
    with st.expander("🔬 View Analysis Methodology"):
        st.markdown(f"""
        **Analysis Process:**
        1. **Signal Acquisition:** Captured 100 samples per panel segment.[cite: 1]
        2. **Feature Engineering:** Calculated 5 key structural indicators (Mean, Std Dev, Min, Max, and CV).[cite: 1]
        3. **Neural Evaluation:** Data is passed through a multi-layer perceptron (MLP) optimized for TFLite.[cite: 1]
        4. **Thresholding:** Signals with a variance/mean profile exceeding 0.5 are flagged as structural anomalies.[cite: 1]
        """)

    if auto_scan:
        time.sleep(delay_map[run_speed])
        st.rerun()