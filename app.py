import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, time

# Always load files relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Fare Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load model & encoders ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load(os.path.join(BASE_DIR, "flight_model.pkl"))
    scaler       = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    airline_enc  = joblib.load(os.path.join(BASE_DIR, "Airline_encoder.pkl"))
    source_enc   = joblib.load(os.path.join(BASE_DIR, "Source_encoder.pkl"))
    dest_enc     = joblib.load(os.path.join(BASE_DIR, "Destination_encoder.pkl"))
    stops_enc    = joblib.load(os.path.join(BASE_DIR, "Total_Stops_encoder.pkl"))
    info_enc     = joblib.load(os.path.join(BASE_DIR, "Additional_Info_encoder.pkl"))
    return model, scaler, airline_enc, source_enc, dest_enc, stops_enc, info_enc

model, scaler, airline_enc, source_enc, dest_enc, stops_enc, info_enc = load_artifacts()

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-size: 1.1rem; font-weight: 600;
        width: 100%; margin-top: 10px;
    }
    .stButton > button:hover { opacity: 0.9; transform: scale(1.01); }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 15px;
        text-align: center; margin-top: 1.5rem;
    }
    .result-box h1 { font-size: 2.8rem; margin: 0; }
    .result-box p  { font-size: 1.1rem; margin: 0.3rem 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## ✈️ Flight Fare Predictor")
st.markdown("Fill in your flight details below to get an **instant price estimate**.")
st.markdown("---")

# ── Input Form ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("🛫 Airline", sorted(airline_enc.classes_.tolist()))
    source  = st.selectbox("📍 Source (From)", sorted(source_enc.classes_.tolist()))
    dest    = st.selectbox("📍 Destination (To)", sorted(dest_enc.classes_.tolist()))
    stops   = st.selectbox("🔁 Total Stops", sorted(stops_enc.classes_.tolist()))
    info    = st.selectbox("ℹ️ Additional Info", sorted(info_enc.classes_.tolist()))

with col2:
    journey_date = st.date_input("📅 Date of Journey", value=date.today())
    dep_time     = st.time_input("🕐 Departure Time", value=time(10, 0))
    arr_time     = st.time_input("🕑 Arrival Time",   value=time(14, 30))

    dur_col1, dur_col2 = st.columns(2)
    with dur_col1:
        dur_hr  = st.number_input("⏱ Duration (hrs)",  min_value=0, max_value=24, value=4)
    with dur_col2:
        dur_min = st.number_input("⏱ Duration (mins)", min_value=0, max_value=59, value=30)

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("💰 Predict Fare"):
    if source == dest:
        st.warning("⚠️ Source and Destination cannot be the same!")
    else:
        try:
            input_data = {
                "Airline":         airline_enc.transform([airline])[0],
                "Source":          source_enc.transform([source])[0],
                "Destination":     dest_enc.transform([dest])[0],
                "Total_Stops":     stops_enc.transform([stops])[0],
                "Additional_Info": info_enc.transform([info])[0],
                "j_date":          journey_date.day,
                "j_mon":           journey_date.month,
                "a_hr":            arr_time.hour,
                "a_min":           arr_time.minute,
                "d_hr":            dep_time.hour,
                "d_min":           dep_time.minute,
                "du_hu":           dur_hr,
                "du_min":          dur_min,
            }

            input_df     = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction   = model.predict(input_scaled)[0]

            st.markdown(f"""
            <div class="result-box">
                <p>Estimated Flight Fare</p>
                <h1>₹ {prediction:,.0f}</h1>
                <p>{airline} · {source} → {dest} · {stops}</p>
            </div>
            """, unsafe_allow_html=True)

            # Details expander
            with st.expander("📋 See Input Summary"):
                summary = pd.DataFrame({
                    "Field": ["Airline","From","To","Stops","Journey Date",
                              "Departure","Arrival","Duration","Extra Info"],
                    "Value": [airline, source, dest, stops,
                              journey_date.strftime("%d %b %Y"),
                              dep_time.strftime("%H:%M"),
                              arr_time.strftime("%H:%M"),
                              f"{dur_hr}h {dur_min}m", info]
                })
                st.table(summary)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest Regressor · Trained on Indian domestic flight data · R² ≈ 0.86")
