"""
Local ETL pipeline.

Raw CSV -> clean data -> processed Parquet -> curated ML-ready CSV
"""

import os
import json
import pandas as pd
from data_quality import run_data_quality_checks


RAW_PATH = "data/raw/synthetic_admissions.csv"
PROCESSED_PATH = "data/processed/admissions_processed.parquet"
CURATED_PATH = "data/curated/readmission_features.csv"
QUALITY_REPORT_PATH = "data/processed/data_quality_report.json"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    df = df.drop_duplicates()
    df = df[df["age"].between(18, 100)]
    df = df[df["length_of_stay"] > 0]

    return df


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gender_encoded"] = df["gender"].map({"Male": 1, "Female": 0})

    diagnosis_dummies = pd.get_dummies(df["diagnosis"], prefix="diagnosis", dtype=int)

    feature_cols = [
        "admission_id",
        "patient_id",
        "age",
        "gender_encoded",
        "previous_admissions",
        "length_of_stay",
        "medication_count",
        "lab_abnormal_count",
        "icu_stay",
        "readmission_30_days"
    ]

    curated_df = pd.concat([df[feature_cols], diagnosis_dummies], axis=1)

    return curated_df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/curated", exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    quality_report = run_data_quality_checks(df)
    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(quality_report, f, indent=4)

    clean_df = clean_data(df)
    clean_df.to_parquet(PROCESSED_PATH, index=False)

    curated_df = build_ml_features(clean_df)
    curated_df.to_csv(CURATED_PATH, index=False)

    print("ETL completed successfully.")
    print(f"Processed data saved to: {PROCESSED_PATH}")
    print(f"Curated ML dataset saved to: {CURATED_PATH}")
    print(f"Data quality report saved to: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
