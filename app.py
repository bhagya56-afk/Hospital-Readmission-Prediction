import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ======================================================================
# PAGE CONFIGURATION & THEME RE-ARCHITECTING
# ======================================================================
st.set_page_config(
    page_title="Pulse Clinical Analytics Console",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection to change the UI/UX from the dark grid to a modern, crisp clean EHR interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* App Clean Layout Card Styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 4px 12px;
        border-radius: 50px;
    }
    
    .status-pill.online {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    /* Clean Typography */
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #0f172a;
        margin-bottom: 4px;
    }
    
    .dashboard-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 24px;
    }
    
    .section-banner {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# CORE CONSTANTS — (Must remain untouched for model pipeline alignment)
# ======================================================================
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-pioglitazone",
]
DRUG_LEVELS = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
RAW_BINARY_DRUG_COLS = ["examide", "citoglipton"]

AGE_GROUPS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
GENDER_OPTIONS = ["Female", "Male"]

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}
DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within institution to Medicare swing bed",
    16: "Discharged/transferred/referred to another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home (Medicaid hospice)",
    20: "Expired in a medical facility (Medicaid hospice)",
    21: "Expired, place unknown (Medicaid hospice)",
    22: "Discharged/transferred to another rehab facility",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility (Medicaid only)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred/referred to a psychiatric hospital",
    29: "Discharged/transferred to a Critical Access Hospital",
    30: "Discharged/transferred to another type of health care institution",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    15: "Not Available", 17: "NULL", 18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 20: "Not Mapped",
    21: "Unknown/Invalid", 22: "Transfer from hospital inpatient (separate claim)",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}

DIAG_CHOICES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes",
    "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]

CATEGORY_LEVELS = {
    "race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"],
    "gender": ["Female", "Male", "Unknown/Invalid"],
    "diag_1": sorted(DIAG_CHOICES),
    "diag_2": sorted(DIAG_CHOICES),
    "diag_3": sorted(DIAG_CHOICES),
    "change": ["Ch", "No"],
    "diabetesMed": ["No", "Yes"],
    "examide": ["No"],
    "citoglipton": ["No"],
}
for _drug in DRUG_COLS:
    CATEGORY_LEVELS[_drug] = ["0", "1", "2", "3"]


# ======================================================================
# DATA RESOURCE PIPELINE LOADERS
# ======================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
    artifacts_loaded = True
    load_error = None
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


# ======================================================================
# INFERENCE SYSTEM COMPUTATIONS
# ======================================================================
def age_group_to_midpoint(age_group: str) -> float:
    nums = np.array(re.findall(r"\d+", age_group), dtype=float)
    return float(nums.mean())


def encode_drug(value: str) -> int:
    return DRUG_LEVELS.get(value, 0)


def build_raw_input_row(fv: dict) -> pd.DataFrame:
    row = {}
    row["age"] = age_group_to_midpoint(fv["age_group"])
    row["admission_type_id"] = fv["admission_type_id"]
    row["discharge_disposition_id"] = fv["discharge_disposition_id"]
    row["admission_source_id"] = fv["admission_source_id"]
    row["time_in_hospital"] = fv["time_in_hospital"]
    row["num_lab_procedures"] = fv["num_lab_procedures"]
    row["num_procedures"] = fv["num_procedures"]
    row["num_medications"] = fv["num_medications"]
    row["number_outpatient"] = fv["number_outpatient"]
    row["number_emergency"] = fv["number_emergency"]
    row["number_inpatient"] = fv["number_inpatient"]
    row["number_diagnoses"] = fv["number_diagnoses"]

    row["total_utilization"] = (
        fv["number_outpatient"] + fv["number_emergency"] + fv["number_inpatient"]
    )

    row["diag_1"] = fv["diag_1"]
    row["diag_2"] = fv["diag_2"]
    row["diag_3"] = fv["diag_3"]
    row["race"] = fv["race"]
    row["gender"] = fv["gender"]
    row["change"] = fv["change"]
    row["diabetesMed"] = fv["diabetesMed"]

    for col in DRUG_COLS:
        row[col] = str(encode_drug(fv.get(col, "No")))

    for col in RAW_BINARY_DRUG_COLS:
        row[col] = fv.get(col, "No")

    return pd.DataFrame([row])


def preprocess_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = raw_df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        if col in CATEGORY_LEVELS:
            raw_df[col] = pd.Categorical(raw_df[col], categories=CATEGORY_LEVELS[col])

    encoded_df = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)
    input_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
    return input_df


def predict_readmission(fv: dict):
    raw_df = build_raw_input_row(fv)
    input_df = preprocess_for_model(raw_df)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]
    return int(prediction[0]), float(probability[0])


# ======================================================================
# LAYOUT STRUCTURE REMODEL: TWO-COLUMN ANALYSIS LAYOUT
# ======================================================================

# Top App Header Bar Area
col_header, col_badge = st.columns([0.8, 0.2])
with col_header:
    st.markdown('<div class="dashboard-title">Pulse Risk Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Decision Support Console for 30-Day Diabetic Patient Readmission Risks</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown('<div style="text-align: right; margin-top: 12px;"><span class="status-pill online">● ENGINE ONLINE</span></div>', unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"⚠️ Initialization Error: Missing core telemetry pipeline file artifacts. ({load_error})")
    st.stop()

# Layout Generation Split
input_workspace, output_workspace = st.columns([0.45, 0.55], gap="large")

with input_workspace:
    st.markdown('<div class="section-banner">Patient Intake Form</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("Demographics & History")
        c1, c2, c3 = st.columns(3)
        with c1:
            race = st.selectbox("Race Base", RACE_OPTIONS)
        with c2:
            gender = st.selectbox("Gender Orientation", GENDER_OPTIONS)
        with c3:
            age_group = st.selectbox("Cohort Age Bracket", AGE_GROUPS, index=6)
            
        st.subheader("Administrative Log")
        admission_type_label = st.selectbox("Encounter Admission Stream", list(ADMISSION_TYPE_MAP.values()))
        admission_type_id = [k for k, v in ADMISSION_TYPE_MAP.items() if v == admission_type_label][0]
        
        admission_source_label = st.selectbox("Inbound Referral Stream Source", list(ADMISSION_SOURCE_MAP.values()), index=6)
        admission_source_id = [k for k, v in ADMISSION_SOURCE_MAP.items() if v == admission_source_label][0]
        
        discharge_disposition_label = st.selectbox("Planned Discharge Exit Strategy", list(DISCHARGE_DISPOSITION_MAP.values()))
        discharge_disposition_id = [k for k, v in DISCHARGE_DISPOSITION_MAP.items() if v == discharge_disposition_label][0]

        time_in_hospital = st.slider("Encounter Duration Scale (Days In Bed)", 1, 14, 3)

        st.subheader("Clinical Activity Totals")
        t1, t2, t3 = st.columns(3)
        with t1:
            num_lab_procedures = st.number_input("Lab Profiles Executed", 0, 150, 40)
            number_outpatient = st.number_input("Outpatient Visits (12mo)", 0, 50, 0)
        with t2:
            num_procedures = st.number_input("Surgical/Other Interventions", 0, 10, 1)
            number_emergency = st.number_input("ER Admissions (12mo)", 0, 50, 0)
        with t3:
            num_medications = st.number_input("Formulary Drug Count", 0, 80, 15)
            number_inpatient = st.number_input("Prior Inpatient Stays", 0, 30, 0)
            
        number_diagnoses = st.slider("Total Indexed Secondary Diagnoses", 1, 16, 7)

        st.subheader("Diagnostic Categorization")
        d1, d2, d3 = st.columns(3)
        with d1:
            diag_1 = st.selectbox("Primary ICD Vector (Diag 1)", DIAG_CHOICES, index=3)
        with d2:
            diag_2 = st.selectbox("Secondary Vector (Diag 2)", DIAG_CHOICES, index=0)
        with d3:
            diag_3 = st.selectbox("Tertiary Vector (Diag 3)", DIAG_CHOICES, index=0)

        st.subheader("Therapeutic Formulations")
        drug_values = {}
        primary_drugs = ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride", "repaglinide"]
        
        # Grid loop layout substitution for a more concise UI presentation
        for chunk in [primary_drugs[i:i + 4] for i in range(0, len(primary_drugs), 4)]:
            cols = st.columns(4)
            for idx, drug_name in enumerate(chunk):
                with cols[idx]:
                    drug_values[drug_name] = st.selectbox(drug_name.capitalize(), ["No", "Steady", "Up", "Down"], key=f"v_{drug_name}")
                    
        with st.expander("Secondary Sub-therapeutic Agent Tracks"):
            remaining_drugs = [d for d in DRUG_COLS if d not in primary_drugs]
            for chunk in [remaining_drugs[i:i + 4] for i in range(0, len(remaining_drugs), 4)]:
                cols = st.columns(4)
                for idx, drug_name in enumerate(chunk):
                    if idx < len(cols):
                        with cols[idx]:
                            drug_values[drug_name] = st.selectbox(drug_name.replace("-", " ").title(), ["No", "Steady", "Up", "Down"], key=f"v_{drug_name}")

        m1, m2 = st.columns(2)
        with m1:
            change = st.selectbox("Course Dosages Adjusted During Visit", ["No", "Ch"])
        with m2:
            diabetesMed = st.selectbox("Active Insulin/Diabetes Treatment Plan", ["Yes", "No"])

# ======================================================================
# ANALYTICS WORKSPACE GENERATION
# ======================================================================
with output_workspace:
    st.markdown('<div class="section-banner">Dynamic Readmission Diagnostic Panel</div>', unsafe_allow_html=True)
    
    # Pack parameters
    form_values = {
        "race": race, "gender": gender, "age_group": age_group,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "change": change, "diabetesMed": diabetesMed,
        **drug_values,
    }

    try:
        prediction, probability = predict_readmission(form_values)
        risk_pct = probability * 100
        safe_pct = 100 - risk_pct
        confidence = "High Accuracy Profile" if abs(risk_pct - 50) > 20 else "Intermediate Variance Profile"

        # Change styles depending dynamically on output context
        if prediction == 1:
            theme_color = "#dc2626" # Clean Crimson Red
            alert_header = "🚨 ELEVATED READMISSION HAZARD DETECTED"
            alert_body = "Patient telemetry charts closely track historical high-risk re-entry patterns within 30-days post-discharge. Immediate clinical workflow mitigation and structured oversight tracking recommended."
        else:
            theme_color = "#2563eb" # Modern Clinical Blue
            alert_header = "✅ PROFILE STABILIZED"
            alert_body = "The structural characteristics of this inpatient event signify baseline risk compliance patterns. Standard institutional post-discharge workflow patterns remain acceptable."

        # High visibility clean KPI Metric Cards
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 5px solid {theme_color}; padding: 20px; border-radius: 8px; margin-bottom: 24px;">
            <h4 style="margin: 0 0 6px 0; color: {theme_color}; font-size: 16px; letter-spacing: -0.2px;">{alert_header}</h4>
            <p style="margin: 0; color: #475569; font-size: 13.5px; line-height: 1.5;">{alert_body}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Visualizations Minimal Configuration Block
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 44, "color": "#0f172a", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": theme_color, "thickness": 0.25},
                "bgcolor": "#f1f5f9",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1"
            }
        ))
        
        fig.update_layout(
            height=200, 
            margin=dict(l=30, r=30, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Dual Segment Breakdown Display Elements
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight:600; color: #64748b; text-transform: uppercase;">Uncomplicated Baseline Prob.</div>
                <div style="font-size: 24px; font-weight:700; color: #0f172a; margin-top:4px;">{safe_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight:600; color: #64748b; text-transform: uppercase;">Model Categorization Profile</div>
                <div style="font-size: 16px; font-weight:600; color: #334155; margin-top:10px;">{confidence}</div>
            </div>
            """, unsafe_allow_html=True)

        # Indeterminate safety validation tracking boundary indicators
        if 45 <= risk_pct <= 55:
            st.warning("⚠️ Equivocal Risk Warning: Patient metrics map to an intermediate clinical threshold border profile. Augment evaluation via qualitative baseline history verification.")

        st.caption("🔒 Computational Support Logic Only — This analysis framework functions as a screening evaluation heuristic and does not supplant final professional clinical directives.")

    except Exception as e:
        st.error(f"Inference Pipeline Error: Could not evaluate input tracking array. Detailed trace: {e}")

# Footer Signature Element Blocks
st.markdown("""
<div style="margin-top: 80px; padding-top: 16px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 11px; color: #94a3b8;">
    Pulse Analytical Framework • Institutional Strategy Testing Platform Prototype Only • Not Validated for Independent Patient Diagnostics
</div>
""", unsafe_allow_html=True)
