# Dashboard Upgrade Instructions

Replace this file in your project:

```text
src/dashboard/app.py
```

with the new `app.py` file from this ZIP.

Then run:

```powershell
streamlit run src\dashboard\app.py
```

After testing successfully, commit and push:

```powershell
git add src/dashboard/app.py
git commit -m "Upgrade dashboard with patient risk prediction"
git push
```
