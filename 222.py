import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("XGB.pkl")

# Streamlit config
st.set_page_config(
    page_title="Blood-based model for predicting PPI after hip fracture",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 Blood-based model for predicting PPI after hip fracture")
st.write("Enter CBC and inflammatory markers to predict the risk of post-operative persistent inflammation (PPI).")

# Feature names
features = [
    'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
    'LYMPH%', 'MONO%', 'NEUT%', 'EO%', 'BASO%',
    'LYMPH#', 'MONO#', 'NEUT#', 'EO#', 'BASO#',
    'RDW-CV', 'PDW', 'MPV', 'PCT', 'P-LCR', 'CRP'
]

# Class labels
result_labels = {0: "Hip fracture", 1: "PPI"}

def predict(input_data):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0]
    return result_labels[pred[0]], prob

# Input panel
st.subheader("Patient Laboratory Input")

cols = st.columns(3)
feature_values = {}

for idx, feature in enumerate(features):
    col = cols[idx % 3]
    feature_values[feature] = col.number_input(feature, value=1.0, format="%.3f")

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    if all(val is not None for val in feature_values.values()):
        input_array = np.array([[feature_values[f] for f in features]]).astype(float)
        predicted_class, probabilities = predict(input_array)

        st.success(f"Prediction: **{predicted_class}**")

        # Bar plot
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["Hip fracture", "PPI"], probabilities)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        st.write(f"• Hip fracture probability: **{probabilities[0]*100:.2f}%**")
        st.write(f"• PPI probability: **{probabilities[1]*100:.2f}%**")

    else:
        st.warning("Please fill in all feature values before predicting.")
