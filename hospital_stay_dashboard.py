import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap
import numpy as np
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import os
from datetime import datetime

# =============================
# 1️⃣ Page Setup & Theme
# =============================
st.set_page_config(page_title="Hospital BMS", layout="wide", page_icon="🏥")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("assets/style.css")

# Custom Header
st.markdown("""
<div class="main-header">
    <div style="font-size: 2rem;">🏥</div>
    <div>
        <h1 class="brand-title">Hospital BMS</h1>
        <p class="brand-subtitle">ML and Data Privacy Project | Real-time Bed Management & Predictions</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# 2️⃣ Load Resources & State
# =============================
@st.cache_resource
def load_model():
    try:
        data = joblib.load("best_hospital_stay_model_comprehensive.pkl")
        return data
    except Exception as e:
        return None

@st.cache_data
def load_background_data():
    try:
        df = pd.read_csv("healthcare_dataset_comprehensive.csv")
        X = df.drop(columns=['Stay_Days', 'Long_Stay', 'Patient_ID'], errors='ignore')
        return X.sample(20, random_state=42)
    except:
        return None

import json

BED_STORAGE = "bed_storage.json"

def save_bed_state():
    with open(BED_STORAGE, "w") as f:
        json.dump(st.session_state.beds, f)

def load_bed_state():
    if os.path.exists(BED_STORAGE):
        with open(BED_STORAGE, "r") as f:
            return json.load(f)
    return None

# Initialize Bed State
if "beds" not in st.session_state:
    persisted_beds = load_bed_state()
    if persisted_beds:
        st.session_state.beds = persisted_beds
    else:
        # Default 50 Beds if no file exists
        beds = [{"id": i+1, "status": "Free", "patient": None, "type": "General"} for i in range(50)]
        st.session_state.beds = beds
        save_bed_state()

def add_new_bed():
    new_id = len(st.session_state.beds) + 1
    st.session_state.beds.append({"id": new_id, "status": "Free", "patient": None, "type": "General"})
    save_bed_state()
    st.toast(f"New Bed {new_id} added successfully!", icon="🛏️")

def remove_last_bed():
    if len(st.session_state.beds) > 0:
        removed = st.session_state.beds.pop()
        save_bed_state()
        st.toast(f"Bed {removed['id']} removed.", icon="🗑️")

def toggle_bed_status(bed_index):
    current = st.session_state.beds[bed_index]["status"]
    if current == "Free":
        st.session_state.beds[bed_index]["status"] = "Cleaning"
    elif current == "Cleaning":
        st.session_state.beds[bed_index]["status"] = "Free"
    elif current == "Occupied":
        # Discharge logic: Occupied -> Cleaning
        st.session_state.beds[bed_index]["status"] = "Cleaning"
        st.session_state.beds[bed_index]["patient"] = None
        st.toast(f"Bed {bed_index+1} discharged and sent for cleaning!", icon="🧹")
    save_bed_state()

def admit_patient(bed_id, patient_name):
    for bed in st.session_state.beds:
        if bed["id"] == bed_id:
            bed["status"] = "Occupied"
            bed["patient"] = patient_name
            save_bed_state()
            return True
    return False

# Load Logic
model_data = load_model()
if isinstance(model_data, dict):
    model = model_data["model"]
    metadata = model_data.get("metadata", {})
else:
    model = model_data
    metadata = {}

# =============================
# 3️⃣ Global Actions (Sidebar)
# =============================
with st.sidebar:
    st.markdown("### 🛠️ Global Controls")
    st.info("Directly manage the system's persistent data and state.")
    
    if st.checkbox("🔓 Enable Reset Operations"):
        if st.button("🗑️ Reset ALL Data", help="Permanently delete all patient records and reset all beds to Free.", type="primary"):
            # 1. Delete History CSV
            if os.path.exists("prediction_history.csv"):
                os.remove("prediction_history.csv")
            
            # 2. Delete Bed Storage JSON
            if os.path.exists(BED_STORAGE):
                os.remove(BED_STORAGE)
            
            # 3. Reset Session State
            st.session_state.beds = [{"id": i+1, "status": "Free", "patient": None, "type": "General"} for i in range(50)]
            if "last_prediction" in st.session_state:
                del st.session_state.last_prediction
            
            # 4. Save clean state
            save_bed_state()
            
            st.toast("System Wiped Successfully!", icon="♻️")
            st.rerun()
    
    st.markdown("---")
    st.caption("v2.1.0 | Data Privacy Mode Active")

# =============================
# 4️⃣ Top Navigation (Tabs)
# =============================
tabs = st.tabs(["⚡ Dashboard", "🛏️ Bed Management", "🔮 Predict Stay", "📂 Patient Records", "📊 Analytics"])

# =============================
# 🏠 TAB 1: DASHBOARD
# =============================
with tabs[0]:
    st.markdown("## Dashboard")
    st.markdown("Real-time hospital bed management and predictions")
    
    # ---------------------------
    # 1. Real Metrics Logic
    # ---------------------------
    # Bed Metrics
    total_beds = len(st.session_state.beds)
    occupied = len([b for b in st.session_state.beds if b["status"] == "Occupied"])
    available = len([b for b in st.session_state.beds if b["status"] == "Free"])
    cleaning = len([b for b in st.session_state.beds if b["status"] == "Cleaning"])
    occ_rate = (occupied / total_beds) * 100
    
    # Patient Metrics form History
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        # Handle column name compatibility
        col_name = "Predicted_Stay_Type" if "Predicted_Stay_Type" in df_hist.columns else "Predicted_Stay"
        
        if col_name in df_hist.columns:
            total_patients = len(df_hist)
            # Count "Long" in the value (matches "Long Stay (>7 days)" and "Long")
            long_stay_count = len(df_hist[df_hist[col_name].astype(str).str.contains("Long", na=False)])
            short_stay_count = total_patients - long_stay_count
        else:
            total_patients = 0
            long_stay_count = 0
            short_stay_count = 0
    else:
        total_patients = 0
        long_stay_count = 0
        short_stay_count = 0

    # ---------------------------
    # 2. Metric Cards
    # ---------------------------
    m1, m2, m3, m4 = st.columns(4)
    
    def metric_card(label, value, subtext, border_color):
        st.markdown(f"""
        <div class="metric-card {border_color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

    with m1: metric_card("Total Patients", str(total_patients), "Records analyzed", "border-blue")
    with m2: metric_card("Capacity", str(total_beds), f"{occupied} Occupied", "border-orange")
    with m3: metric_card("Available", str(available), "Beds ready", "border-teal")
    with m4: metric_card("Bed Occupancy", f"{occ_rate:.0f}%", f"{cleaning} in Cleaning", "border-red")

    st.markdown("---")
    
    # ---------------------------
    # 3. Model & Feature Analytics
    # ---------------------------
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown('<div class="white-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
        
        # Dynamic Model Card
        if metadata:
            auc = metadata.get('auc_score', 0.0)
            model_date = metadata.get('training_date', 'Unknown')
            # Infer model name or default
            model_name = metadata.get('model_type', "Voting Ensemble (Best)") 
        else:
            auc = 0.0
            model_date = "N/A"
            model_name = "Not Loaded"

        st.markdown(f"""
        <div style="background: #ECFDF5; padding: 1.5rem; border-radius: 8px; border: 1px solid #A7F3D0; margin-bottom: 1rem;">
            <div style="color: #065F46; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Active Model</div>
            <div style="color: #064E3B; font-size: 1.5rem; font-weight: 800; margin: 4px 0;">{model_name}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
                <span style="background:white; padding:4px 8px; border-radius:4px; color:#059669; font-weight:bold; font-size:0.9rem;">AUC: {auc:.4f}</span>
                <span style="font-size:0.8rem; color:#047857;">{model_date.split(' ')[0]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("The active model is automatically selected based on the highest validation ROC-AUC score.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Trends (Live)</div>', unsafe_allow_html=True)
        
        if total_patients > 0:
            # Pie Chart with Explicit Color Mapping
            trend_df = pd.DataFrame({
                "Stay Type": ["Long Stay", "Short Stay"],
                "Count": [long_stay_count, short_stay_count]
            })
            
            fig = px.pie(trend_df, 
                         names="Stay Type", 
                         values="Count",
                         hole=0.6, 
                         color="Stay Type",
                         color_discrete_map={"Long Stay": "#EF4444", "Short Stay": "#00A86B"})
                         
            fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No prediction history available. Go to 'Predict Stay' to generate data.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🛏️ TAB 2: BED MANAGEMENT
# =============================
with tabs[1]:
    st.markdown("## 🛏️ Bed Management")
    
    # Bed Configuration Controls
    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        if st.button("➕ Add New Bed", help="Add a new bed to the hospital system"):
            add_new_bed()
            st.rerun()
    with bc2:
        if st.button("🗑️ Remove Last Bed", help="Remove the last bed if it's not needed", type="secondary"):
            remove_last_bed()
            st.rerun()
    
    st.info("Click on a bed number to toggle status: Free ↔️ Cleaning. Occupied beds allows discharge.")
    st.markdown("---")
    
    cols = st.columns(10)
    for i, bed in enumerate(st.session_state.beds):
        status = bed["status"]
        if status == "Free":
            icon = "✅ Free"
            style_col = ":green"
            help_text = "Click to mark for Cleaning"
        elif status == "Occupied":
            icon = f"🔴 {bed['patient']}"
            style_col = ":red"
            help_text = "Click to Discharge"
        else: # Cleaning
            icon = "🧹 Cleaning"
            style_col = ":orange"
            help_text = "Click to mark as Free"

        # Display bed
        with cols[i % 10]:
            st.markdown(f"**Bed {bed['id']}**")
            if st.button(f"Manage {bed['id']}", key=f"bed_{i}", help=f"{icon} - {help_text}", use_container_width=True):
                 toggle_bed_status(i)
                 st.rerun()
            
            if status == "Free":
                 st.markdown(f"{style_col}[{icon}]")
            elif status == "Occupied":
                 st.markdown(f"{style_col}[Occ.]")
                 st.caption(f"{bed['patient']}")
            else:
                 st.markdown(f"{style_col}[Clean]")

# =============================
# 🔮 TAB 3: PREDICT STAY
# =============================
with tabs[2]:
    st.markdown("## 🔮 Predict Patient Stay")
    
    # Diagnosis Mapping
    diagnosis_map = {
        'Cardiology': ['Heart Failure', 'Angina', 'Coronary Artery Disease'],
        'Neurology': ['Stroke', 'Migraine', 'Epilepsy'],
        'Orthopedics': ['Knee Replacement', 'Hip Fracture', 'Back Pain'],
        'Oncology': ['Chemotherapy', 'Tumor Surgery', 'Radiotherapy'],
        'Pediatrics': ['Viral Infection', 'Asthma Attack', 'Dehydration'],
        'Gastroenterology': ['Appendicitis', 'Gastritis', 'Pancreatitis'],
        'Gynecology': ['C-Section', 'Normal Delivery', 'Hysterectomy'],
        'Pulmonology': ['Pneumonia', 'COPD', 'Bronchitis']
    }

    c_left, c_right = st.columns([1, 2], gap="large")
    
    with c_left:
        st.markdown('<div class="white-box">', unsafe_allow_html=True)
        st.subheader("📝 New Admission Form")
        
        # New Patient Inputs
        p_name = st.text_input("Patient Name", "New Patient")
        
        st.markdown("**1. Demographics & Insurance**")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 0, 100, 45)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        insurance = st.selectbox("Insurance Type", ["Private", "Medicare", "Medicaid", "Uninsured"])

        st.markdown("**2. Admission Details**")
        c3, c4 = st.columns(2)
        dept = c3.selectbox("Department", list(diagnosis_map.keys()))
        adm_type = c4.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
        ward = st.selectbox("Ward Type", ["General", "Private", "ICU", "HDU"])
        deposit = st.number_input("Admission Deposit ($)", 0, 20000, 7000, 500)
        
        st.markdown("**3. Clinical Condition**")
        diagnosis = st.selectbox("Diagnosis", diagnosis_map.get(dept, ["Other"]))
        
        severity = st.slider("Severity Score", 1, 5, 3)
        lab_abn = st.slider("Lab Abnormality Score", 0, 4, 1)
        
        st.markdown("---")
        submit = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        if submit: 
            if model:
                # Prepare Data
                input_data = {
                    "Age": [age], "Gender": [gender], "Admission_Type": [adm_type],
                    "Insurance_Type": [insurance], "Department": [dept], "Diagnosis": [diagnosis],
                    "Ward_Type": [ward], "Severity_Score": [severity], "Num_Comorbidities": [1], 
                    "Prev_Hospitalizations": [0], "Procedure_Complexity": [2],
                    "Lab_Abnormality_Score": [lab_abn], "Visitors_Count": [2],
                    "Blood_Sugar_Level": [110], "Admission_Deposit": [deposit]
                }
                single_df = pd.DataFrame(input_data)
                
                try:
                    prob_raw = model.predict_proba(single_df)[0][1]
                    prob = prob_raw * 100
                    is_long = prob > 50 
                    
                    st.session_state.last_prediction = {
                        "p_name": p_name,
                        "prob": prob,
                        "is_long": is_long,
                        "df": single_df
                    }
                    
                    # Logic to save to CSV
                    df_history = single_df.copy()
                    df_history["Patient_Name"] = p_name
                    df_history["Predicted_Stay_Type"] = "Long Stay" if is_long else "Short Stay"
                    df_history["Probability"] = prob
                    df_history["Prediction_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df_history.to_csv("prediction_history.csv", mode='a', header=not os.path.exists("prediction_history.csv"), index=False)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        # Show Prediction if exists
        if "last_prediction" in st.session_state:
            pred = st.session_state.last_prediction
            res_color = "#EF4444" if pred["is_long"] else "#059669"
            res_icon = "⚠️" if pred["is_long"] else "✅"
            res_title = "High Risk: Long Stay (>7 Days)" if pred["is_long"] else "Low Risk: Short Stay (≤7 Days)"
            
            st.markdown(f"""
            <div style="background: white; padding: 2rem; border-radius: 16px; border: 2px solid {res_color}; text-align: center;">
                <div style="font-size: 3rem;">{res_icon}</div>
                <h2 style="color: {res_color}; margin: 0;">{res_title}</h2>
                <div style="font-size: 2.5rem; font-weight: 800; color: {res_color};">{pred['prob']:.1f}%</div>
                <p>Confidence Level for Extended Care</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🛏️ Quick Bed Allocation")
            free_beds = [b["id"] for b in st.session_state.beds if b["status"] == "Free"]
            
            if free_beds:
                col_sel, col_con = st.columns([3, 1])
                sel_bed = col_sel.selectbox("Assign to Bed", free_beds)
                if col_con.button("Confirm Allocation", type="primary"):
                    admit_patient(sel_bed, pred["p_name"])
                    del st.session_state.last_prediction # Clear after allocation
                    st.toast(f"Success! {pred['p_name']} assigned to Bed {sel_bed}")
                    st.rerun()
            else:
                st.warning("⚠️ No beds available for allocation.")
        else:
            st.info("👈 Fill out the patient form and click 'Run Prediction' to see results.")

# =============================
# 📂 TAB 4: PATIENT RECORDS
# =============================
with tabs[3]:
    st.markdown("## 📂 Patient Records")
    
    # Global Reset Control (Always Available)
    grc1, grc2 = st.columns([3, 1])
    with grc2:
        if st.button("🗑️ Reset All Beds/Records", help="Irreversibly deletes all patient records and resets all beds."):
            if os.path.exists("prediction_history.csv"): os.remove("prediction_history.csv")
            if os.path.exists(BED_STORAGE): os.remove(BED_STORAGE)
            st.session_state.beds = [{"id": i+1, "status": "Free", "patient": None, "type": "General"} for i in range(50)]
            save_bed_state()
            st.toast("System Reset complete!", icon="♻️")
            st.rerun()

    if os.path.exists("prediction_history.csv"):
        hist_df = pd.read_csv("prediction_history.csv")
        if "Prediction_Date" in hist_df.columns:
            hist_df = hist_df.sort_values(by="Prediction_Date", ascending=False)
        
        # Stats
        r1, r2 = st.columns(2)
        r1.metric("Total Records", len(hist_df))
        r2.metric("Recent Additions", len(hist_df[pd.to_datetime(hist_df["Prediction_Date"]) > (datetime.now() - pd.Timedelta(hours=1))]))
        
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No patient records in database. Beds can still be reset if needed.")

# =============================
# 📊 TAB 5: ANALYTICS
# =============================
with tabs[4]:
    st.markdown("## 📊 Analytics")
    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])
    if uploaded_file and model:
        df = pd.read_csv(uploaded_file)
        if st.button("Run Global SHAP"):
            with st.spinner("Analyzing..."):
                bg = load_background_data()
                if bg is not None:
                    explainer = shap.KernelExplainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=bg.columns))[:, 1], bg)
                    shap_values = explainer.shap_values(df[bg.columns].head(20))
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, df[bg.columns].head(20), plot_type="bar", show=False)
                    st.pyplot(fig)

# End of Dashboard
