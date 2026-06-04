"""
Train baseline readmission prediction model.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = "data/curated/readmission_features.csv"
MODEL_PATH = "models/readmission_model.joblib"


def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["readmission_30_days", "admission_id", "patient_id"])
    y = df["readmission_30_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
