import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------------- CONFIG ----------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model (1).pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessing_tools.pkl"

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ---------------------- LOAD ARTIFACTS ----------------------
@st.cache_resource
def load_model_and_preprocessor():
    # Load CatBoost model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load preprocessing pipeline/transformer
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    if not hasattr(preprocessor, "transform"):
        raise ValueError("Preprocessing object must support .transform()")

    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# ---------------------- FEATURES ----------------------
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ---------------------- APP LAYOUT ----------------------
st.markdown("<h1 style='text-align:center;color:#1f77b4'>Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("Fill in the details of the customer below to predict churn probability.")

# ---- Input Form ----
with st.form("customer_form"):
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ['Female', 'Male'])
        SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        Partner = st.selectbox("Partner", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['No', 'Yes'])
        PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
        MultipleLines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
        InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])

    with col2:
        OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        TechSupport = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
        Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        PaymentMethod = st.selectbox(
            "Payment Method", 
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        )

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=800.0)

    submitted = st.form_submit_button("Predict Churn")

# ---- Prediction ----
if submitted:
    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_dict])

    try:
        # Transform using the pipeline / ColumnTransformer
        processed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]

        # Display results
        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Customer is likely to **CHURN** with probability {probability:.2f}")
        else:
            st.success(f"✅ Customer is **NOT likely to churn** with probability {1-probability:.2f}")

        st.markdown("**Submitted Data:**")
        st.table(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
