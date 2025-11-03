import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("XGB.pkl")

# Page config
st.set_page_config(
    page_title="Blood-based model for predicting PPI after hip fracture",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 Blood-based model for predicting PPI after hip fracture")
st.write("Enter CBC and inflammatory markers to predict the risk of post-operative persistent inflammation (PPI).")

# Feature list with units and default values
feature_info = {
    "WBC (10^9/L)": 23.3,
    "RBC (10^12/L)": 2.94,
    "HGB (g/L)": 82,
    "HCT (%)": 25,
    "MCV (fL)": 85.3,
    "MCH (pg)": 28,
    "MCHC (g/L)": 328,
    "PLT (10^9/L)": 513,
    "LYMPH% (%)": 12.2,
    "MONO% (%)": 8.4,
    "NEUT% (%)": 74.2,
    "EO% (%)": 5,
    "BASO% (%)": 0.2,
    "LYMPH# (10^9/L)": 2.84,
    "MONO# (10^9/L)": 1.96,
    "NEUT# (10^9/L)": 17.29,
    "EO# (10^9/L)": 1.16,
    "BASO# (10^9/L)": 0.05,
    "RDW-CV (%)": 13.1,
    "PDW (fL)": 15.9,
    "MPV (fL)": 9.3,
    "PCT (%)": 0.48,
    "P-LCR (%)": 21.7,
    "CRP (mg/L)": 120.29
}

features = list(feature_info.keys())
result_labels = {0: "Hip fracture", 1: "PPI"}

def predict(input_data):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0]
    return result_labels[pred[0]], prob

# Input area
st.subheader("Patient Laboratory Values")
cols = st.columns(3)
feature_values = {}

for idx, (feature, default) in enumerate(feature_info.items()):
    col = cols[idx % 3]
    feature_values[feature] = col.number_input(feature, value=float(default), format="%.3f")

# Predict button
if st.button("🔍 Predict", use_container_width=True):

    input_array = np.array([[feature_values[f] for f in features]]).astype(float)
    predicted_class, probabilities = predict(input_array)

    st.success(f"Prediction: **{predicted_class}**")

    # Probability Display
    st.subheader("Prediction Probability")

    fig, ax = plt.subplots(figsize=(3, 3))  # Smaller chart
    ax.bar(["Hip fracture", "PPI"], probabilities, width=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Outcome Probability")

    st.pyplot(fig)

    st.write(f"• Hip fracture probability: **{probabilities[0]*100:.2f}%**")
    st.write(f"• PPI probability: **{probabilities[1]*100:.2f}%**")
