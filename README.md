# Healthcare Patient Readmission Prediction Platform

## Overview

An end-to-end Healthcare Analytics and Machine Learning platform designed to predict 30-day patient readmission risk.

This project demonstrates Data Engineering and Data Science skills using:

- Python ETL Pipelines
- AWS S3 Data Lake
- AWS Glue Data Catalog
- Amazon Athena
- Machine Learning
- Streamlit Dashboard
- Docker (Phase 3)

---

## Architecture

Healthcare Admissions Data
        ↓
Python ETL Pipeline
        ↓
Amazon S3 Data Lake
        ↓
AWS Glue Data Catalog
        ↓
Amazon Athena
        ↓
Machine Learning Model
        ↓
Streamlit Dashboard

---

## Key Features

- Synthetic healthcare admissions data generation
- Data quality validation
- Feature engineering
- Readmission prediction model
- Interactive analytics dashboard
- AWS cloud integration

---

## Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 78% |
| Precision | 79% |
| Recall | 80% |
| F1 Score | 80% |
| ROC-AUC | 0.86 |

---

## Technologies

- Python
- Pandas
- Scikit-Learn
- Streamlit
- AWS S3
- AWS Glue
- Amazon Athena
- GitHub

---

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python src/etl/generate_synthetic_data.py
python src/etl/run_local_etl.py
python src/models/train_model.py

streamlit run src/dashboard/app.py
