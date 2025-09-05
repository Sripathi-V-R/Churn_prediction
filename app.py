import streamlit as st
import joblib
import pandas as pd

# -------------------
# Load model + preprocessor safely
# -------------------
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    # Check if preprocessor has .transform() method
    if not hasattr(preprocessor, "transform"):
        # If it's a dict, raise a clear error
        raise ValueError(
            f"Preprocessor is of type {type(preprocessor)}. "
            "It must be a scikit-learn transformer or Pipeline with .transform()."
        )

    return model, preprocessor


# -------------------
# Main app
# -------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below to predict whether they are likely to churn.")

# Load model and preprocessor
try:
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"❌ Failed to load model/preprocessor: {e}")
    st.stop()

# -------------------
# Input fields
# -------------------
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

    submitted = st.form_submit_button("Predict Churn")

# -------------------
# Prediction
# -------------------
if submitted:
    try:
        # Put inputs in DataFrame
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges
        }])

        # Preprocess input
        X_processed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn. Probability: {proba:.2f}")
        else:
            st.success(f"✅ Customer is not likely to churn. Probability: {proba:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
