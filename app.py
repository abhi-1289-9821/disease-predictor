import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Disease Predictor", layout="wide")

# --------------------------
# Custom CSS (PRO UI)
# --------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
}
h1, h2, h3, h4, h5 {
    color: white;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: #1e293b;
    margin-bottom: 15px;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.small-text {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load model
# --------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# --------------------------
# Header
# --------------------------
st.markdown("""
<h1 style='text-align:center;'>🩺 Disease Prediction System</h1>
<p style='text-align:center;' class='small-text'>
AI-powered symptom-based disease prediction
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# Layout
# --------------------------
col1, col2 = st.columns([2, 1])

# --------------------------
# LEFT → INPUT
# --------------------------
with col1:
    st.markdown("<h3>Select Symptoms</h3>", unsafe_allow_html=True)

    search = st.text_input("🔍 Search symptoms")

    input_data = {}

    filtered = [f for f in features if search.lower() in f.lower()]

    cols = st.columns(3)

    for i, f in enumerate(filtered):
        with cols[i % 3]:
            input_data[f] = st.checkbox(f)

# --------------------------
# RIGHT → OUTPUT
# --------------------------
with col2:
    st.markdown("<h3>Prediction</h3>", unsafe_allow_html=True)

    if st.button("Predict Disease"):

        for f in features:
            if f not in input_data:
                input_data[f] = 0

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        probs = model.predict_proba(input_scaled)
        top3 = np.argsort(probs, axis=1)[:, -3:]
        classes = model.classes_

        st.markdown("<p class='small-text'>Top predictions</p>", unsafe_allow_html=True)

        for i, idx in enumerate(reversed(top3[0]), 1):
            prob = probs[0][idx] * 100

            st.markdown(f"""
            <div class="card">
                <h4>#{i} {classes[idx]}</h4>
                <p class="small-text">Confidence: {prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            #d:\disease-predictor\.venv\Scripts\python.exe -m streamlit run app.py