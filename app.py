import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------------- CONFIG ----------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model (1).pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessing_tools.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ---------------------- LOAD ARTIFACTS ----------------------
@st.cache_resource
def load_model_and_preprocessor():
    # Load CatBoost model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load preprocessing dictionary
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    if not isinstance(preprocessor, dict):
        raise ValueError("Expected preprocessing_tools.pkl to be a dict with 'scaler' and 'encoder'")

    scaler = preprocessor.get("scaler")
    encoder = preprocessor.get("encoder")

    if scaler is None or encoder is None:
        raise ValueError("Scaler or encoder missing in preprocessing_tools.pkl")

    return model, scaler, encoder

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Load everything
model, scaler, encoder = load_model_and_preprocessor()
df = load_data()

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
st.markdown("Filter a customer from the dataset and predict churn.")

# ---------------------- FILTERS ----------------------
st.subheader("Filter Customer")
filtered_df = df.copy()

for col in categorical_cols + numeric_cols:
    if col in categorical_cols:
        unique_vals = filtered_df[col].unique().tolist()
        selected = st.selectbox(f"Filter {col}", ["All"] + unique_vals, key=col)
        if selected != "All":
            filtered_df = filtered_df[filtered_df[col] == selected]
    else:
        min_val = float(filtered_df[col].min())
        max_val = float(filtered_df[col].max())
        val_range = st.slider(f"{col} range", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[(filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])]

st.markdown(f"Filtered **{len(filtered_df)}** customers")
st.dataframe(filtered_df.reset_index(drop=True))

# ---------------------- SELECT ROW FOR PREDICTION ----------------------
st.subheader("Select a customer row for prediction")
if not filtered_df.empty:
    selected_index = st.number_input("Row index", min_value=0, max_value=len(filtered_df)-1, value=0, step=1)
    input_row = filtered_df.iloc[[selected_index]]
    st.markdown("Selected customer data:")
    st.table(input_row)

    # ---------------------- PREDICTION ----------------------
    if st.button("Predict Churn for Selected Customer"):
        try:
            # Numeric features
            num_data = input_row[numeric_cols].values
            scaled_num = scaler.transform(num_data)

            # Categorical features
            cat_data = input_row[categorical_cols].values
            encoded_cat = encoder.transform(cat_data)

            # Combine
            processed_input = np.hstack([scaled_num, encoded_cat])

            # Predict
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ Customer is likely to **CHURN** with probability {probability:.2f}")
            else:
                st.success(f"✅ Customer is **NOT likely to churn** with probability {1-probability:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("No customers match the filter criteria.")

