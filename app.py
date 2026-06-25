import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ======================================================================
# CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="Pulse Readmission Dashboard",
    page_icon="🩺",
    layout="centered", # Better readability than wide-stretch for standard use
)

# ======================================================================
# CONSTANTS — Must match the notebook training setup exactly
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
# CACHED RESOURCE LOADING
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
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


# ======================================================================
# FEATURE ENGINEERING & PROCESSING
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
# CLEAN, SIMPLIFIED APPLICATION HEADER
# ======================================================================
st.title("🩺 30-Day Hospital Readmission Tool")
st.markdown(
    "Fill out the standard demographic and encounter metrics below to evaluate clinical "
    "readmission risk probability via our validated machine learning backend framework."
)

if not artifacts_loaded:
    st.error(f"⚠️ App files (`model.pkl`, `scaler.pkl`, `feature_columns.pkl`) missing: {load_error}")
    st.stop()

# ======================================================================
# USER-FRIENDLY ORGANIZED INTERACTIVE CONTROLS
# ======================================================================
with st.form("simplified_intake_form"):
    
    st.header("1. Profile & Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        race = st.selectbox("Race Base", RACE_OPTIONS)
    with col2:
        gender = st.selectbox("Gender Identification", GENDER_OPTIONS)
    with col3:
        age_group = st.selectbox("Patient Age Bracket Range", AGE_GROUPS, index=6)

    st.header("2. Encounter Logistics")
    col_adm, col_dis = st.columns(2)
    with col_adm:
        admission_type_label = st.selectbox("Admission Type Designation", list(ADMISSION_TYPE_MAP.values()))
        admission_type_id = [k for k, v in ADMISSION_TYPE_MAP.items() if v == admission_type_label][0]
        
        admission_source_label = st.selectbox("Source Referral Origin", list(ADMISSION_SOURCE_MAP.values()), index=6)
        admission_source_id = [k for k, v in ADMISSION_SOURCE_MAP.items() if v == admission_source_label][0]
    with col_dis:
        discharge_disposition_label = st.selectbox("Discharge Destination Status", list(DISCHARGE_DISPOSITION_MAP.values()))
        discharge_disposition_id = [k for k, v in DISCHARGE_DISPOSITION_MAP.items() if v == discharge_disposition_label][0]
        
        time_in_hospital = st.slider("Duration of Stay (Days)", min_value=1, max_value=14, value=3)

    st.header("3. Clinical Evaluation Metrics")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        num_lab_procedures = st.number_input("Lab Test Count", 0, 150, 40)
        number_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", 0, 50, 0)
    with col_m2:
        num_procedures = st.number_input("Non-Lab Procedures Conducted", 0, 10, 1)
        number_emergency = st.number_input("Prior Emergency Encounters (Past Year)", 0, 50, 0)
    with col_m3:
        num_medications = st.number_input("Total Prescribed Medications", 0, 80, 15)
        number_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", 0, 30, 0)
        
    number_diagnoses = st.slider("Total Diagnoses Entered on Chart", 1, 16, 7)

    st.header("4. Primary Conditions")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        diag_1 = st.selectbox("Primary Diagnosis Class", DIAG_CHOICES, index=3)
    with col_d2:
        diag_2 = st.selectbox("Secondary Diagnosis Class", DIAG_CHOICES, index=0)
    with col_d3:
        diag_3 = st.selectbox("Tertiary Diagnosis Class", DIAG_CHOICES, index=0)

    st.header("5. Diabetic Medication Control Tracks")
    drug_values = {}
    primary_drugs = ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride", "repaglinide"]
    
    # Loop splits common medications into an elegant simple clean layout
    for i in range(0, len(primary_drugs), 4):
        cols = st.columns(4)
        for idx, drug in enumerate(primary_drugs[i:i+4]):
            with cols[idx]:
                drug_values[drug] = st.selectbox(drug.capitalize(), ["No", "Steady", "Up", "Down"], key=f"inp_{drug}")

    with st.expander("Show Extended Minor Medications Tracks"):
        remaining_drugs = [d for d in DRUG_COLS if d not in primary_drugs]
        for i in range(0, len(remaining_drugs), 4):
            cols = st.columns(4)
            for idx, drug in enumerate(remaining_drugs[i:i+4]):
                with cols[idx]:
                    drug_values[drug] = st.selectbox(drug.replace("-", " ").title(), ["No", "Steady", "Up", "Down"], key=f"inp_{drug}")

    col_ch, col_med = st.columns(2)
    with col_ch:
        change = st.selectbox("Medication Dosage Change Noted", ["No", "Ch"])
    with col_med:
        diabetesMed = st.selectbox("Any Diabetes Meds Active on Plan", ["Yes", "No"])

    st.markdown("---")
    submitted = st.form_submit_button("Calculate Patient Risk Index", use_container_width=True)

# ======================================================================
# ACCESSIBLE DATA INTERPRETATION OUTPUT
# ======================================================================
if submitted:
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
        confidence = "HIGH PROFILE CONVERGENCE" if abs(risk_pct - 50) > 20 else "MODERATE STABILITY SHIFT"

        st.subheader("Analysis Results Summary")

        # Standard clean alerts based on model classification thresholds
        if prediction == 1:
            st.error("🚨 **High Readmission Risk Alert**")
            verdict_text = "The patient's chart tracks closely with historical patients who were readmitted within 30 days."
        else:
            st.success("✅ **Low Readmission Risk Confirmed**")
            verdict_text = "The patient's clinical profile matches stable baselines with lower re-entry tendencies."

        # High visibility statistics without using complex custom layouts
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric(label="Readmission Probability", value=f"{risk_pct:.1f}%")
        with col_res2:
            st.metric(label="Stable Recovery Profile", value=f"{safe_pct:.1f}%")
        with col_res3:
            st.metric(label="Pipeline Confidence Check", value=confidence)

        st.info(f"**Interpretation Detail:** {verdict_text}")

        # Basic accessibility fallback safety message 
        if 45 <= risk_pct <= 55:
            st.warning("⚠️ **Borderline Assessment Profile Warning:** Risk values are hanging close to standard equilibrium thresholds. Defer heavily to hands-on qualitative screening.")

    except Exception as e:
        st.error(f"Calculation Interrupted by Pipeline Event: {e}")
else:
    st.info("Input clinical details above and select **Calculate Patient Risk Index** to run the prediction model.")
