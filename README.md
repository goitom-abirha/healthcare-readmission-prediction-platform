# Healthcare Patient Readmission Prediction Platform

## Phase 2 Added
This version includes:
- Synthetic healthcare admissions dataset generator
- Local ETL pipeline
- Data quality checks
- Feature engineering starter
- Curated dataset output

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/etl/generate_synthetic_data.py
python src/etl/run_local_etl.py
streamlit run src/dashboard/app.py
```
