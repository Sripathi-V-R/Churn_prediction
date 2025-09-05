import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model (1).pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessing_tools.pkl"
DATA_RAW_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"   # before encoding
DATA_PROCESSED_PATH = BASE_DIR / "TelcoChurn_Processed.csv" # after encoding

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_model_and_preprocessor():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_RAW_PATH)
    processed = pd.read_csv(DATA_PROCESSED_PATH)
    return raw, processed


def apply_preprocessing(preprocessor, input_df, processed_columns):
    """Apply preprocessing and align columns to match training dataset."""
    # Case 1: pipeline
    if hasattr(preprocessor, "transform"):
        transformed = preprocessor.transform(input_df)
        if isinstance(transformed, np.ndarray):
            df_out = pd.DataFrame(transformed, columns=processed_columns)
        else:
            df_out = pd.DataFrame(transformed)
        return df_out

    # Case 2: dict
    if isinstance(preprocessor, dict):
        df = input_df.copy()

        # Encoder
        if "encoder" in preprocessor and hasattr(preprocessor["encoder"], "transform"):
            df = pd.DataFrame(
                preprocessor["encoder"].transform(df),
                columns=preprocessor["encoder"].get_feature_names_out(),
            )

        # Scaler
        if "scaler" in preprocessor and hasattr(preprocessor["scaler"], "transform"):
            numeric_cols = preprocessor.get("numeric_cols", [])
            df[numeric_cols] = preprocessor["scaler"].transform(df[numeric_cols])

        # Align columns with training
        for col in processed_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[processed_columns]

        return df

    raise ValueError("❌ Preprocessor format not recognized")


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- Header ----
    st.markdown(
        f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        model, preprocessor = load_model_and_preprocessor()
        raw_data, processed_data = load_data()
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app predicts whether a telecom customer is likely to churn.")

    # Input
    st.subheader("📋 Enter Customer Details")
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(raw_data.columns):
        with cols[i % 3]:
            if raw_data[col].dtype == "object":
                val = st.selectbox(col, options=sorted(raw_data[col].dropna().unique()))
            else:
                val = st.number_input(
                    col, value=float(raw_data[col].dropna().median()), step=1.0
                )
            input_data[col] = val

    # Predict
    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            processed = apply_preprocessing(preprocessor, input_df, processed_data.columns)

            pred = model.predict(processed)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(processed)[0][1]

            st.markdown("### ✅ Prediction Result")
            if pred == 1:
                st.error("🚨 This customer is **likely to churn**.")
            else:
                st.success("💚 This customer is **not likely to churn**.")
            if prob is not None:
                st.info(f"Churn Probability: **{prob:.2%}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
