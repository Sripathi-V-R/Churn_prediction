import streamlit as st
import joblib
import pandas as pd

# -------------------
# Load model + preprocessor
# -------------------
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    if not hasattr(preprocessor, "transform"):
        raise ValueError("Preprocessor must be a scikit-learn transformer with .transform()")

    return model, preprocessor

# -------------------
# Streamlit App
# -------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Fill in customer details to predict churn probability.")

# Load model
try:
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"❌ Failed to load model/preprocessor: {e}")
    st.stop()

# Example fields — adjust to match your dataset
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    try:
        # Input as DataFrame
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Transform
        X_processed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer likely to churn (probability {proba:.2f})")
        else:
            st.success(f"✅ Customer unlikely to churn (probability {proba:.2f})")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
