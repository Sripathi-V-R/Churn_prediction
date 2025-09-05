import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model (1).pkl"   # <-- exact name
PREPROCESSOR_PATH = BASE_DIR / "preprocessing_tools.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_model_and_preprocessor():
    # Check files exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"❌ Preprocessor file not found at {PREPROCESSOR_PATH}")

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load preprocessing tools
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    # Ensure preprocessor supports transform()
    if not hasattr(preprocessor, "transform"):
        raise ValueError("Preprocessor object must support .transform()")

    return model, preprocessor


@st.cache_data
def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Data file not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- Header ----
    header_cols = st.columns([1, 3, 1])
    with header_cols[1]:
        st.markdown(
            f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Load artifacts
    try:
        model, preprocessor = load_model_and_preprocessor()
        raw_data = load_raw_data()
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        st.stop()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app predicts whether a telecom customer is likely to churn.")
        st.warning("⚠️ Note: This is a demo ML app, not a business recommendation.")

    # ---- Input Section ----
    st.subheader("📋 Enter Customer Details")

    input_data = {}
    cols = st.columns(3)
    all_inputs = {}

    for i, col in enumerate(raw_data.columns):
        with cols[i % 3]:
            if raw_data[col].dtype == "object":
                val = st.selectbox(col, options=sorted(raw_data[col].dropna().unique()))
            else:
                val = st.number_input(
                    col,
                    value=float(raw_data[col].dropna().median()),
                    step=1.0,
                )
            all_inputs[col] = val

    # ---- Predict Button ----
    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([all_inputs])

            # Apply preprocessing
            processed = preprocessor.transform(input_df)

            # Prediction
            pred = model.predict(processed)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(processed)[0][1]

            # ---- Result ----
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
