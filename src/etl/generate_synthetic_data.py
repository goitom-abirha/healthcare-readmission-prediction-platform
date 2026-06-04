"""
Generate synthetic healthcare admissions data for readmission prediction.

This allows the project to start immediately without waiting for MIMIC-IV access.
Later, this file can be replaced with real healthcare data ingestion.
"""

import os
import numpy as np
import pandas as pd


def generate_admissions_data(n_records: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    patient_ids = np.random.randint(10000, 20000, size=n_records)
    age = np.random.randint(18, 95, size=n_records)
    gender = np.random.choice(["Male", "Female"], size=n_records)
    diagnosis = np.random.choice(
        ["Diabetes", "Heart Failure", "Pneumonia", "COPD", "Kidney Disease", "Hypertension"],
        size=n_records
    )

    previous_admissions = np.random.poisson(lam=1.2, size=n_records)
    length_of_stay = np.random.randint(1, 21, size=n_records)
    medication_count = np.random.randint(1, 18, size=n_records)
    lab_abnormal_count = np.random.randint(0, 10, size=n_records)
    icu_stay = np.random.choice([0, 1], size=n_records, p=[0.82, 0.18])

    risk_score = (
        0.02 * age
        + 0.7 * previous_admissions
        + 0.25 * length_of_stay
        + 0.2 * medication_count
        + 0.35 * lab_abnormal_count
        + 1.2 * icu_stay
    )

    probability = 1 / (1 + np.exp(-(risk_score - 8)))
    readmission_30_days = np.random.binomial(1, probability)

    admission_dates = pd.date_range(start="2025-01-01", periods=n_records, freq="h")
    discharge_dates = admission_dates + pd.to_timedelta(length_of_stay, unit="D")

    df = pd.DataFrame({
        "admission_id": range(1, n_records + 1),
        "patient_id": patient_ids,
        "age": age,
        "gender": gender,
        "diagnosis": diagnosis,
        "previous_admissions": previous_admissions,
        "length_of_stay": length_of_stay,
        "medication_count": medication_count,
        "lab_abnormal_count": lab_abnormal_count,
        "icu_stay": icu_stay,
        "admission_date": admission_dates,
        "discharge_date": discharge_dates,
        "readmission_30_days": readmission_30_days
    })

    return df


if __name__ == "__main__":
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    df = generate_admissions_data()
    output_path = os.path.join(output_dir, "synthetic_admissions.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved synthetic dataset to: {output_path}")
    print(df.head())
