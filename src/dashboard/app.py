"""
Streamlit dashboard for healthcare readmission prediction.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Healthcare Readmission Prediction Dashboard",
    layout="wide"
)

st.title("Healthcare Patient Readmission Prediction Dashboard")

curated_path = "data/curated/readmission_features.csv"

if os.path.exists(curated_path):
    df = pd.read_csv(curated_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Admissions", len(df))
    col2.metric("Readmission Cases", int(df["readmission_30_days"].sum()))
    col3.metric("Readmission Rate", f"{df['readmission_30_days'].mean() * 100:.2f}%")

    st.subheader("Sample Curated Dataset")
    st.dataframe(df.head(20))

    st.subheader("Readmission Distribution")
    st.bar_chart(df["readmission_30_days"].value_counts())

    st.subheader("Average Risk Features by Readmission Status")
    st.dataframe(
        df.groupby("readmission_30_days")[
            ["age", "previous_admissions", "length_of_stay", "medication_count", "lab_abnormal_count", "icu_stay"]
        ].mean()
    )
else:
    st.warning("Curated dataset not found. Run the ETL pipeline first.")
    st.code("python src/etl/generate_synthetic_data.py\npython src/etl/run_local_etl.py")
