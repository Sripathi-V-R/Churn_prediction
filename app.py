import streamlit as st
import pandas as pd
import pickle

# ---- Load Model ----
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")

st.write("""
This app predicts whether a person is likely to have diabetes based on input health parameters.
""")

# Input fields
st.sidebar.header("Patient Information")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Use model directly (CatBoost/XGBoost/RandomForest handle raw input)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"The model predicts: **Diabetic** (Risk: {prediction_proba*100:.2f}%)")
    else:
        st.success(f"The model predicts: **Not Diabetic** (Risk: {prediction_proba*100:.2f}%)")
