import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('XGB.pkl')

st.set_page_config(
    page_title="RB Screening Model",
    page_icon="🩺",
    layout="wide"
)

# --- Header ---
st.markdown("""
# 🩺 Retinoblastoma (RB) Blood-Based Screening
### AI-assisted screening based on routine blood indicators
""")

# Feature list
features = [
    'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
    'LYMPH%', 'MONO%', 'NEUT%', 'EO%', 'BASO%',
    'LYMPH#', 'MONO#', 'NEUT#', 'EO#', 'BASO#',
    'RDW-CV', 'PDW', 'MPV', 'PCT', 'P-LCR', 'CRP'
]

result_labels = {0: 'normal', 1: 'RB'}

# Predict function
def predict_and_show_results(data):
    p = model.predict(data)
    pro = model.predict_proba(data)
    return result_labels[p[0]], {result_labels[i]:pro[0][i] for i in range(len(pro[0]))}

# --- Layout Design ---
with st.container():
    st.markdown("### 🧪 Enter Laboratory Indicators")
    col1, col2, col3 = st.columns([1,1,1])

    inputs = {}

    with col1:
        st.subheader("CBC Basics")
        inputs['WBC'] = st.number_input('WBC (10^9/L)', value=6.5)
        inputs['RBC'] = st.number_input('RBC (10^12/L)', value=4.5)
        inputs['HGB'] = st.number_input('HGB (g/L)', value=130.0)
        inputs['HCT'] = st.number_input('HCT (%)', value=40.0)
        inputs['PLT'] = st.number_input('PLT (10^9/L)', value=250.0)

        st.subheader("Platelets")
        inputs['PDW'] = st.number_input('PDW (fL)', value=10.0)
        inputs['MPV'] = st.number_input('MPV (fL)', value=10.0)
        inputs['PCT'] = st.number_input('PCT (%)', value=0.2)
        inputs['P-LCR'] = st.number_input('P-LCR (%)', value=30.0)

    with col2:
        st.subheader("RBC Indices")
        inputs['MCV'] = st.number_input('MCV (fL)', value=90.0)
        inputs['MCH'] = st.number_input('MCH (pg)', value=30.0)
        inputs['MCHC'] = st.number_input('MCHC (g/L)', value=330.0)
        inputs['RDW-CV'] = st.number_input('RDW-CV (%)', value=13.0)

        st.subheader("Differential %")
        inputs['LYMPH%'] = st.number_input('LYMPH% (%)', value=35.0)
        inputs['MONO%'] = st.number_input('MONO% (%)', value=8.0)
        inputs['NEUT%'] = st.number_input('NEUT% (%)', value=55.0)
        inputs['EO%'] = st.number_input('EO% (%)', value=2.0)
        inputs['BASO%'] = st.number_input('BASO% (%)', value=0.5)

    with col3:
        st.subheader("Differential #")
        inputs['LYMPH#'] = st.number_input('LYMPH#', value=2.0)
        inputs['MONO#'] = st.number_input('MONO#', value=0.5)
        inputs['NEUT#'] = st.number_input('NEUT#', value=3.5)
        inputs['EO#'] = st.number_input('EO#', value=0.1)
        inputs['BASO#'] = st.number_input('BASO#', value=0.03)

        st.subheader("Inflammation Marker")
        inputs['CRP'] = st.number_input('CRP (mg/L)', value=5.0)

# Predict Button
center = st.columns([1,1,1])[1]
with center:
    if st.button("🔍 Screening", use_container_width=True):
        arr = np.array([[inputs[f] for f in features]])
        pred, prob = predict_and_show_results(arr)

        st.success(f"Prediction: **{pred}**")
        st.write("### Probability:")
        for k,v in prob.items(): st.write(f"{k}: {v:.4f} ({v*100:.2f}%)")




