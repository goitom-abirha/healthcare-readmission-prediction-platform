"""
Data quality checks for healthcare readmission data.
"""

import pandas as pd


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    quality_report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_age_count": int((df["age"] < 0).sum()) if "age" in df.columns else None,
        "invalid_length_of_stay_count": int((df["length_of_stay"] <= 0).sum()) if "length_of_stay" in df.columns else None,
    }

    return quality_report
