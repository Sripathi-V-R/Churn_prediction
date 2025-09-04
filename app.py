# === Diabetes Prediction App (Single Page PDF + Charts + Logo) ===
# Save as: Diabetes_Prediction.py
# Run: py -m streamlit run Diabetes_Prediction.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import tempfile
import os
from datetime import datetime
from pathlib import Path
import atexit

# ---------------------- CONFIG ----------------------
MODEL_PATH = Path(r"C:/Users/sripathivr/Downloads/lightgbm_best_model.pkl")
SCALER_PATH = Path(r"C:/Users/sripathivr/Downloads/scaler.pkl")
DATA_PATH  = Path(r"C:/Users/sripathivr/Downloads/preprocessed_dataset.csv")

ASSETS_DIR = Path(r"C:/Users/sripathivr/Tasks/Diabetes_Prediction/assets")
LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.png"

APP_TITLE = "🏥 Diabetes Prediction Medical Portal"
PRIMARY_COLOR = "#7b1e1e"

HEALTHY_RANGES = {
    "Age (yrs)": (20, 50),
    "BMI": (18.5, 24.9),
    "HbA1c (%)": (4.0, 5.6),
    "Glucose (mg/dL)": (90, 140)
}

_TEMP_FILES = []
def _register_temp(path): _TEMP_FILES.append(path)
def _cleanup_temp():
    for p in _TEMP_FILES:
        try: os.remove(p)
        except: pass
atexit.register(_cleanup_temp)

# ---------------------- LOAD ARTIFACTS ----------------------
@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)

    model, scaler = None, None
    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("estimator") or loaded.get("model_object")
        scaler = loaded.get("scaler") or loaded.get("preprocessor")
        if model is None and "pipeline" in loaded: model = loaded["pipeline"]
    else:
        model = loaded

    if scaler is None and SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    if scaler is None:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            expected_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
            present = [c for c in expected_cols if c in df.columns]
            if len(present) < 4:
                raise ValueError("Dataset does not contain required columns to fit a scaler. Provide a saved scaler.")
            scaler = StandardScaler().fit(df[present])
        else:
            raise FileNotFoundError("Scaler not found and DATA_PATH not available.")

    return model, scaler

# ---------------------- CHART HELPERS ----------------------
def draw_range_bars(values_dict):
    params = list(values_dict.keys())
    vals = [values_dict[k] for k in params]

    fig, ax = plt.subplots(figsize=(6, 3))
    y_pos = np.arange(len(params))
    colors = []
    for i, p in enumerate(params):
        low, high = HEALTHY_RANGES.get(p, (0, np.max([vals[i],1])))
        colors.append("tab:orange" if vals[i]<low or vals[i]>high else "tab:green")

    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=8)
    ax.set_xlabel("Value", fontsize=9)

    for i, p in enumerate(params):
        low, high = HEALTHY_RANGES.get(p,(0,0))
        ax.axvline(low, linestyle="--", linewidth=0.8)
        ax.axvline(high, linestyle="--", linewidth=0.8)
        ax.text(vals[i] if vals[i]<ax.get_xlim()[1] else ax.get_xlim()[1]*0.98, i,
                f" {vals[i]}", va="center", ha="left" if vals[i]<ax.get_xlim()[1]*0.95 else "right", fontsize=8)

    ax.set_title("Health Parameters vs Healthy Ranges", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def draw_radar_chart(values_dict):
    labels = list(values_dict.keys())
    values = [values_dict[k] for k in labels]

    def norm(val,rng):
        low,high = rng; mid=(low+high)/2; span=max(high-low,1e-6)
        return max(0, min(1, 1-min(abs(val-mid)/(span/2),1)))

    ranges_map = {k:HEALTHY_RANGES[k] for k in labels}
    scores = [norm(values[i], ranges_map[labels[i]]) for i in range(len(labels))]

    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    scores_loop = scores+[scores[0]]
    angles_loop = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(3.2,3.2))
    ax = fig.add_subplot(111,polar=True)
    ax.plot(angles_loop,scores_loop,linewidth=1.5)
    ax.fill(angles_loop,scores_loop,alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([0,0.5,1.0])
    ax.set_yticklabels(["Poor","Average","Good"], fontsize=6)
    ax.set_ylim(0,1)
    ax.set_title("Overall Health Profile", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------------- PDF REPORT ----------------------
def generate_medical_pdf(patient,prediction_text,range_bars_png,radar_png):
    pdf = FPDF()
    pdf.add_page()

    # font
    try:
        dejavu = Path("C:/Windows/Fonts/DejaVuSans.ttf")
        if dejavu.exists():
            pdf.add_font("DejaVu","",str(dejavu),uni=True)
            pdf.set_font("DejaVu","",12)
        else:
            pdf.set_font("Arial","",12)
    except: pdf.set_font("Arial","",12)

    # header images
    if LOGO_PATH.exists():
        try: pdf.image(str(LOGO_PATH),10,8,25)
        except: pass
    if BANNER_PATH.exists():
        try: pdf.image(str(BANNER_PATH),140,8,55)
        except: pass

    pdf.set_font_size(16); pdf.set_font(pdf.font_family,"B")
    pdf.cell(0,10,"Diabetes Prediction Medical Report",ln=True,align="C")
    pdf.set_font_size(11); pdf.set_font(pdf.font_family,"")
    pdf.cell(0,6,"City General Hospital",ln=True,align="C")
    pdf.cell(0,6,"123 Health Street, Wellness City  +91-98765-43210",ln=True,align="C")
    pdf.cell(0,6,"Email: care@citygeneralhospital.example",ln=True,align="C")
    pdf.ln(4); pdf.set_draw_color(120,30,30); pdf.set_line_width(0.8); pdf.line(10,45,200,45); pdf.ln(4)

    # patient info
    pdf.set_font(pdf.font_family,"B"); pdf.cell(0,8,"Patient Information",ln=True)
    pdf.set_font(pdf.font_family,"")
    prefix = "Mr." if patient["gender"]=="Male" else "Ms." if patient["gender"]=="Female" else "Mx."
    pdf.multi_cell(0,6,f"Name: {prefix} {patient['name']}\nGender: {patient['gender']}\nAge: {patient['age']} years\nHypertension: {patient['hypertension']}\nHeart Disease: {patient['heart_disease']}\nSmoking History: {patient['smoking_history']}\nReport Date/Time: {patient['timestamp']}")
    pdf.ln(1)

    # measurements
    pdf.set_font(pdf.font_family,"B"); pdf.cell(0,8,"Measurements",ln=True)
    pdf.set_font(pdf.font_family,"")
    col_w=[60,60,60]; row_h=7
    def row(a,b,c): pdf.cell(col_w[0],row_h,a,1); pdf.cell(col_w[1],row_h,b,1); pdf.cell(col_w[2],row_h,c,1); pdf.ln(row_h)
    pdf.set_font(pdf.font_family,"B"); row("Parameter","Value","Healthy Range"); pdf.set_font(pdf.font_family,"")
    row("BMI",f"{patient['bmi']}","18.5 - 24.9"); row("HbA1c (%)",f"{patient['hba1c']}","4.0 - 5.6"); row("Glucose (mg/dL)",f"{patient['glucose']}","90 - 140")
    pdf.ln(2)

    # prediction
    pdf.set_font(pdf.font_family,"B"); pdf.cell(0,8,"Prediction",ln=True)
    pdf.set_font(pdf.font_family,""); pdf.set_text_color(180,0,0 if "Diabetic" in prediction_text else 0); pdf.multi_cell(0,6,prediction_text)
    pdf.set_text_color(0,0,0)

    # charts
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as f1: f1.write(range_bars_png); r_path=f1.name
        with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as f2: f2.write(radar_png); rad_path=f2.name
        pdf.ln(1); pdf.set_font(pdf.font_family,"B"); pdf.cell(0,6,"Visual Summary",ln=True)
        y_img=pdf.get_y()+1; pdf.image(r_path,x=15,y=y_img,w=70); pdf.image(rad_path,x=120,y=y_img,w=70); pdf.ln(55)
    except: pass

    # doctor's recommendation
    pdf.set_font(pdf.font_family,"B"); pdf.cell(0,8,"Doctor's Recommendation",ln=True); pdf.set_font(pdf.font_family,"")
    recommendation = ("WARNING: The model predicts a risk of Diabetes. Please consult a physician for confirmatory testing. Adopt a balanced diet, regular physical activity, and monitor glucose levels closely." if "Diabetic" in prediction_text else "SUCCESS: No diabetes indicated by the model. Maintain a healthy lifestyle and schedule routine check-ups.")
    pdf.multi_cell(0,6,recommendation)

    # footer
    pdf.set_y(-20); pdf.set_font_size(8); pdf.set_font(pdf.font_family,"I")
    pdf.cell(0,6,"This is a computer-generated report and should not replace professional medical advice.",0,1,"C")

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf"); tmp_pdf.close(); pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name,"rb") as f: pdf_bytes=f.read()
    return pdf_bytes

# ---------------------- APP ----------------------
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")
    header_cols = st.columns([1,3,1])
    with header_cols[0]: st.image(str(LOGO_PATH),use_container_width=True) if LOGO_PATH.exists() else None
    with header_cols[1]: st.markdown(f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>",unsafe_allow_html=True)
    with header_cols[2]: st.image(str(BANNER_PATH),use_container_width=True) if BANNER_PATH.exists() else None
    st.markdown("---")

    with st.sidebar: st.header("About"); st.write("Predicts diabetes risk using LightGBM."); st.write("⚠️ Not a medical diagnosis.")

    if "patient" not in st.session_state: st.session_state["patient"] = {}
    if "inputs_ready" not in st.session_state: st.session_state["inputs_ready"] = False

    # Helper for numeric inputs
    def safe_float_input(label, default):
        val = st.text_input(label,value=str(default))
        try: return float(val)
        except ValueError: st.warning(f"Enter a valid number for {label}"); return default

    tab1, tab2, tab3 = st.tabs(["📝 Input","📊 Results","📄 Report"])
    with tab1:
        st.subheader("Enter Patient Details")
        c1,c2=st.columns(2)
        with c1:
            name=st.text_input("Full Name",st.session_state["patient"].get("name","John Doe"))
            gender=st.radio("Gender",("Male","Female","Others"),horizontal=True,index=0 if st.session_state["patient"].get("gender","Male")=="Male" else 1)
            age=safe_float_input("Age (years)",st.session_state["patient"].get("age",30))
            bmi=safe_float_input("BMI",st.session_state["patient"].get("bmi",25.0))
            hba1c=safe_float_input("HbA1c Level (%)",st.session_state["patient"].get("hba1c",5.5))
            glucose=safe_float_input("Blood Glucose (mg/dL)",st.session_state["patient"].get("glucose",120.0))
        with c2:
            hypertension=st.selectbox("Hypertension",("No","Yes"),0 if st.session_state["patient"].get("hypertension","No")=="No" else 1)
            heart_disease=st.selectbox("Heart Disease",("No","Yes"),0 if st.session_state["patient"].get("heart_disease","No")=="No" else 1)
            smoking_history=st.selectbox("Smoking History",("never","current","formerly","No Info","ever","not current"),0)
        st.session_state["patient"]={"name":name.strip(),"gender":gender,"age":float(age),"hypertension":hypertension,"heart_disease":heart_disease,"smoking_history":smoking_history,"bmi":float(bmi),"hba1c":float(hba1c),"glucose":float(glucose),"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state["inputs_ready"]=True; st.success("Inputs updated — go to Results tab.")

    with tab2:
        st.subheader("Prediction & Visualizations")
        if not st.session_state.get("inputs_ready"): st.warning("Complete inputs in Input tab first.")
        else:
            try: model, scaler = load_model_and_scaler()
            except Exception as e: st.error(f"Error loading model/scaler: {e}"); st.stop()
            p = st.session_state["patient"]
            gender_num=1 if p["gender"]=="Male" else 0; hypertension_num=1 if p["hypertension"]=="Yes" else 0; heart_disease_num=1 if p["heart_disease"]=="Yes" else 0
            smoking_map={"never":0,"current":1,"formerly":2,"No Info":3,"ever":4,"not current":5}
            smoking_num=smoking_map.get(p["smoking_history"],3)
            inputs_num=np.array([[p["age"],p["bmi"],p["hba1c"],p["glucose"]]])
            try: scaled=scaler.transform(inputs_num)
            except Exception as e: st.error(f"Scaler transform failed: {e}"); st.stop()
            feature_vector=np.concatenate(([gender_num,hypertension_num,heart_disease_num,smoking_num],scaled.flatten())).reshape(1,-1)
            try:
                if hasattr(model,"predict_proba"): proba=model.predict_proba(feature_vector)[:,1][0]; pred_label=1 if proba>=0.5 else 0; result="Diabetic" if pred_label==1 else "Non-Diabetic"; st.info(f"Model probability (Diabetic): {proba:.2%}")
                else: pred=model.predict(feature_vector); result="Diabetic" if int(pred[0])==1 else "Non-Diabetic"; proba=None
            except Exception as e: st.error(f"Prediction failed: {e}"); st.stop()
            st.error("⚠️ Prediction: **Diabetic**") if result=="Diabetic" else st.success("🎉 Prediction: **Non-Diabetic**")
            values={"Age (yrs)":p["age"],"BMI":p["bmi"],"HbA1c (%)":p["hba1c"],"Glucose (mg/dL)":p["glucose"]}
            colA,colB=st.columns(2)
            with colA: st.caption("Health Parameters vs Healthy Range"); range_png=draw_range_bars(values); st.image(range_png,use_container_width=True)
            with colB: st.caption("Overall Health Profile (Radar)"); radar_png=draw_radar_chart(values); st.image(radar_png,use_container_width=True)
            st.session_state["range_png"]=range_png; st.session_state["radar_png"]=radar_png
            pred_text=f"Prediction: {result}" + (f" (P={proba:.2%})" if proba is not None else ""); st.session_state["prediction_text"]=pred_text

    with tab3:
        st.subheader("Download Report")
        if "prediction_text" not in st.session_state: st.info("Generate prediction first.")
        else:
            pdf_bytes=generate_medical_pdf(st.session_state["patient"],st.session_state["prediction_text"],st.session_state.get("range_png",b""),st.session_state.get("radar_png",b""))
            st.download_button("⬇️ Download Medical PDF",pdf_bytes,"Medical_Report.pdf","application/pdf")
            img=st.session_state.get("range_png",b""); st.download_button("⬇️ Download Range Chart PNG",img,"Health_Ranges.png","image/png") if img else None

if __name__=="__main__": main()
# === End of Program ===
