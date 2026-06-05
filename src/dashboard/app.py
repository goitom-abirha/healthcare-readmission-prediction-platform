import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Healthcare Readmission Prediction Dashboard",
    layout="wide"
)

DATA_PATH = "data/curated/readmission_features.csv"
MODEL_PATH = "models/readmission_model.joblib"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def get_risk_category(probability: float) -> str:
    if probability < 0.35:
        return "Low Risk"
    elif probability < 0.65:
        return "Medium Risk"
    return "High Risk"

def build_input_row(
    age,
    gender,
    previous_admissions,
    length_of_stay,
    medication_count,
    lab_abnormal_count,
    icu_stay,
    diagnosis,
    model_columns,
):
    row = pd.DataFrame(0, index=[0], columns=model_columns)

    if "age" in row.columns:
        row["age"] = age
    if "gender_encoded" in row.columns:
        row["gender_encoded"] = 1 if gender == "Male" else 0
    if "previous_admissions" in row.columns:
        row["previous_admissions"] = previous_admissions
    if "length_of_stay" in row.columns:
        row["length_of_stay"] = length_of_stay
    if "medication_count" in row.columns:
        row["medication_count"] = medication_count
    if "lab_abnormal_count" in row.columns:
        row["lab_abnormal_count"] = lab_abnormal_count
    if "icu_stay" in row.columns:
        row["icu_stay"] = 1 if icu_stay == "Yes" else 0

    diagnosis_col = f"diagnosis_{diagnosis}"
    if diagnosis_col in row.columns:
        row[diagnosis_col] = 1

    return row

st.title("Healthcare Patient Readmission Prediction Dashboard")
st.caption("AWS Data Engineering • Python ETL • Machine Learning • Streamlit Analytics")

df = load_data()
model = load_model()

if df is None:
    st.warning("Curated dataset not found. Run the ETL pipeline first.")
    st.code("python src\\etl\\generate_synthetic_data.py\npython src\\etl\\run_local_etl.py\npython src\\models\\train_model.py")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Data Analytics",
    "Patient Risk Prediction",
    "Project Architecture"
])

with tab1:
    st.subheader("Executive Summary")

    total_admissions = len(df)
    readmission_cases = int(df["readmission_30_days"].sum())
    readmission_rate = df["readmission_30_days"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Admissions", f"{total_admissions:,}")
    col2.metric("Readmission Cases", f"{readmission_cases:,}")
    col3.metric("Readmission Rate", f"{readmission_rate:.2f}%")
    col4.metric("Model ROC-AUC", "0.86")

    st.markdown("### Business Purpose")
    st.write(
        "This dashboard supports healthcare analytics by identifying patients who may be at risk "
        "of 30-day hospital readmission. The platform combines Python ETL, AWS S3 Data Lake, "
        "AWS Glue Data Catalog, Athena SQL, machine learning, and dashboard visualization."
    )

    st.markdown("### Model Performance")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Accuracy", "78%")
    metric_col2.metric("Precision", "79%")
    metric_col3.metric("Recall", "80%")
    metric_col4.metric("F1-Score", "80%")

with tab2:
    st.subheader("Healthcare Data Analytics")

    st.markdown("### Sample Curated Dataset")
    st.dataframe(df.head(25), use_container_width=True)

    st.markdown("### Readmission Distribution")
    readmission_counts = df["readmission_30_days"].value_counts().rename(index={0: "Not Readmitted", 1: "Readmitted"})
    st.bar_chart(readmission_counts)

    st.markdown("### Average Feature Values by Readmission Status")
    feature_cols = [
        "age",
        "previous_admissions",
        "length_of_stay",
        "medication_count",
        "lab_abnormal_count",
        "icu_stay"
    ]
    st.dataframe(
        df.groupby("readmission_30_days")[feature_cols].mean().rename(index={0: "Not Readmitted", 1: "Readmitted"}),
        use_container_width=True
    )

with tab3:
    st.subheader("Patient Readmission Risk Prediction")

    if model is None:
        st.error("Model file not found. Run: python src\\models\\train_model.py")
        st.stop()

    model_columns = list(model.feature_names_in_)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Patient Age", 18, 100, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        previous_admissions = st.slider("Previous Admissions", 0, 10, 2)
        length_of_stay = st.slider("Length of Stay", 1, 30, 5)

    with col2:
        medication_count = st.slider("Medication Count", 1, 25, 6)
        lab_abnormal_count = st.slider("Abnormal Lab Count", 0, 15, 4)
        icu_stay = st.selectbox("ICU Stay", ["No", "Yes"])
        diagnosis = st.selectbox(
            "Primary Diagnosis",
            ["Diabetes", "Heart Failure", "Pneumonia", "COPD", "Kidney Disease", "Hypertension"]
        )

    if st.button("Predict Readmission Risk"):
        input_row = build_input_row(
            age,
            gender,
            previous_admissions,
            length_of_stay,
            medication_count,
            lab_abnormal_count,
            icu_stay,
            diagnosis,
            model_columns,
        )

        probability = model.predict_proba(input_row)[0][1]
        prediction = model.predict(input_row)[0]
        risk_category = get_risk_category(probability)

        st.markdown("### Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Readmission Probability", f"{probability * 100:.2f}%")
        result_col2.metric("Risk Category", risk_category)
        result_col3.metric("Model Prediction", "Readmission" if prediction == 1 else "No Readmission")

        if risk_category == "High Risk":
            st.error("High-risk patient. Recommended action: prioritize follow-up care, medication review, and discharge planning.")
        elif risk_category == "Medium Risk":
            st.warning("Medium-risk patient. Recommended action: schedule follow-up and monitor risk factors.")
        else:
            st.success("Low-risk patient. Recommended action: standard discharge workflow.")

with tab4:
    st.subheader("Project Architecture")

    st.code(
        """
Synthetic Healthcare Data
        ↓
Python ETL Pipeline
        ↓
Raw / Processed / Curated Local Layers
        ↓
Amazon S3 Data Lake
        ↓
AWS Glue Crawler & Data Catalog
        ↓
Amazon Athena SQL Queries
        ↓
Machine Learning Model
        ↓
Streamlit Dashboard
        """,
        language="text"
    )

    st.markdown("### Project Components")
    st.write(
        "- Python ETL for data cleaning, validation, and feature engineering\n"
        "- Amazon S3 Data Lake with raw, processed, and curated zones\n"
        "- AWS Glue Crawler for schema discovery and metadata cataloging\n"
        "- Athena SQL for serverless querying\n"
        "- Random Forest model for 30-day readmission prediction\n"
        "- Streamlit dashboard for analytics and decision support"
    )
