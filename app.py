import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek modern styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Asset Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure model.pkl, scaler.pkl, and feature_columns.pkl are in the same directory.")
        return None, None, None

model, scaler, feature_columns = load_artifacts()

# -----------------------------------------------------------------------------
# 3. Application Layout & Inputs
# -----------------------------------------------------------------------------
st.title("🏥 Patient Readmission Risk Assessment")
st.markdown("Predict the likelihood of a diabetic patient being readmitted within 30 days of discharge.")
st.hr()

# Feature options from your training data mapping
DIAG_CHOICES = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
DRUG_MAPPING = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

# Demo Data Button for quick UX evaluation
if st.button("✨ Load High-Risk Sample Patient Data", key="sample_btn"):
    st.session_state.age = 65.0
    st.session_state.gender = "Male"
    st.session_state.race = "Caucasian"
    st.session_state.time_in_hospital = 7
    st.session_state.num_lab_procedures = 65
    st.session_state.num_procedures = 2
    st.session_state.num_medications = 22
    st.session_state.outpatient = 1
    st.session_state.emergency = 2
    st.session_state.inpatient = 3
    st.session_state.diag_1 = "Circulatory"
    st.session_state.insulin = "Up"
    st.session_state.diabetesMed = "Yes"

# Sidebar: Demographics
st.sidebar.header("👤 Patient Demographics")
age = st.sidebar.slider("Age (Estimated Midpoint)", 5.0, 95.0, st.session_state.get('age', 55.0), step=10.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.get('gender') == "Female" else 1)
race = st.sidebar.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"], index=["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"].index(st.session_state.get('race', 'Caucasian')))

# Main layout split into interactive workflow sections
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Encounter & Clinical Metrics")
    
    tab1, tab2, tab3 = st.tabs(["🏥 Encounter Metadata", "🩻 Diagnostics & History", "💊 Medications"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            admission_type_id = st.number_input("Admission Type ID", min_value=1, max_value=8, value=1)
            admission_source_id = st.number_input("Admission Source ID", min_value=1, max_value=25, value=7)
        with c2:
            discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1, max_value=28, value=1)
            time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, st.session_state.get('time_in_hospital', 4))

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=150, value=st.session_state.get('num_lab_procedures', 40))
            num_procedures = st.number_input("Number of Non-Lab Procedures", min_value=0, max_value=10, value=st.session_state.get('num_procedures', 1))
            num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=st.session_state.get('num_medications', 15))
        with c2:
            outpatient = st.number_input("Prior Outpatient Visits", min_value=0, value=st.session_state.get('outpatient', 0))
            emergency = st.number_input("Prior Emergency Visits", min_value=0, value=st.session_state.get('emergency', 0))
            inpatient = st.number_input("Prior Inpatient Visits", min_value=0, value=st.session_state.get('inpatient', 0))
            
        st.markdown("##### Primary Categorized Diagnoses")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            diag_1 = dishonesty = st.selectbox("Primary (Diag 1)", DIAG_CHOICES, index=DIAG_CHOICES.index(st.session_state.get('diag_1', 'Diabetes')))
        with cc2:
            diag_2 = st.selectbox("Secondary (Diag 2)", DIAG_CHOICES, index=8)
        with cc3:
            diag_3 = st.selectbox("Tertiary (Diag 3)", DIAG_CHOICES, index=8)

    with tab3:
        st.markdown("##### Diabetes Medication Details")
        c1, c2 = st.columns(2)
        with c1:
            change = st.selectbox("Medication Dosage Change?", ["No", "Ch"], index=1 if st.session_state.get('change') == "Ch" else 0)
            diabetesMed = st.selectbox("Prescribed Diabetes Medication?", ["No", "Yes"], index=1 if st.session_state.get('diabetesMed') == "Yes" else 0)
        with c2:
            insulin_val = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=["No", "Steady", "Up", "Down"].index(st.session_state.get('insulin', 'No')))
        
        # Initialize remaining standard drugs to 0 ('No') for simplified UI layout
        drug_status = {d: 0 for d in [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
            'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
            'metformin-pioglitazone'
        ]}
        # Explicitly map chosen insulin status
        drug_status['insulin'] = DRUG_MAPPING[insulin_val]

# -----------------------------------------------------------------------------
# 4. Feature Re-engineering & Matching Encoding Pipeline
# -----------------------------------------------------------------------------
# Assemble raw record mirroring your notebook state
raw_data = {
    'age': age,
    'admission_type_id': admission_type_id,
    'discharge_disposition_id': discharge_disposition_id,
    'admission_source_id': admission_source_id,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': outpatient,
    'number_emergency': emergency,
    'number_inpatient': inpatient,
    'total_utilization': outpatient + emergency + inpatient,
    'race': race,
    'gender': gender,
    'diag_1': diag_1,
    'diag_2': diag_2,
    'diag_3': diag_3,
    'change': change,
    'diabetesMed': diabetesMed,
    'metformin-rosiglitazone': 'No'  # constant handled in notebook clean up
}
raw_data.update(drug_status)

input_df = pd.DataFrame([raw_data])

# Encode input record matching pandas dummy structure exactly
encoded_input = pd.get_dummies(input_df)

# Align columns to perfectly match feature_columns.pkl trained dimensions
final_features = pd.DataFrame(0, index=[0], columns=feature_columns)
for col in final_features.columns:
    if col in encoded_input.columns:
        final_features[col] = encoded_input[col].values

# -----------------------------------------------------------------------------
# 5. Prediction Engine Output Display
# -----------------------------------------------------------------------------
with col2:
    st.subheader("🔮 Assessment Output")
    
    if model is not None:
        # Scale continuous/engineered variables
        scaled_features = scaler.transform(final_features)
        
        # Run inference
        prob = model.predict_proba(scaled_features)[0][1]
        pred = 1 if prob >= 0.5 else 0
        
        # Display customized Gauge Metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Readmission Risk Score", value=f"{prob*100:.1f}%")
        st.progress(float(prob))
        st.markdown('</div>', unsafe_allow_html=True)
        
        if pred == 1:
            st.error("⚠️ **High Risk Alert:** Patient is highly likely to be readmitted within 30 days. Actionable intervention or alternative discharge care workflows recommended.")
        else:
            st.success("✅ **Low Risk Profile:** Patient is unlikely to be readmitted within 30 days according to current clinical markers.")
            
        with st.expander("🔍 System Technical Details"):
            st.write("Aligned Feature Space Columns Vector Length:", final_features.shape[1])
            st.dataframe(final_features.loc[:, (final_features != 0).any(axis=0)])