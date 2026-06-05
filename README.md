# Healthcare Patient Readmission Prediction Platform

## Overview

An end-to-end Healthcare Analytics, Data Engineering, and Machine Learning platform designed to predict 30-day patient readmission risk.

This project demonstrates cloud-based data engineering and machine learning using AWS services and modern analytics tools.

---

## Architecture

Healthcare Admissions Data
↓
Python ETL Pipeline
↓
Amazon S3 Data Lake (Raw Zone)
↓
AWS Glue Crawler
↓
AWS Glue Data Catalog
↓
AWS Glue Visual ETL (PySpark)
↓
Amazon S3 Curated Zone (Parquet)
↓
Amazon Athena SQL
↓
Machine Learning Model
↓
Streamlit Dashboard

---

## Key Features

* Synthetic healthcare admissions data generation
* Data quality validation and cleansing
* Feature engineering pipeline
* AWS S3 Data Lake architecture
* AWS Glue Crawler and Data Catalog integration
* AWS Glue Visual ETL transformations
* Amazon Athena SQL analytics
* Patient readmission risk prediction
* Interactive Streamlit dashboard
* End-to-end cloud analytics workflow

---

## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 78%   |
| Precision | 79%   |
| Recall    | 80%   |
| F1 Score  | 80%   |
| ROC-AUC   | 0.86  |

---

## Technologies

### Data Engineering

* Python
* AWS S3
* AWS Glue Crawler
* AWS Glue Data Catalog
* AWS Glue Visual ETL
* Amazon Athena

### Data Science

* Pandas
* Scikit-Learn
* Feature Engineering
* Classification Modeling

### Visualization

* Streamlit

### DevOps & Version Control

* Git
* GitHub

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
```

---

## Business Impact

This platform helps healthcare organizations identify patients at risk of readmission, enabling proactive interventions, improving patient outcomes, and reducing healthcare costs.
