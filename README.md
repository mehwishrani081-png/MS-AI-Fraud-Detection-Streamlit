# MS-AI-Fraud-Detection-Streamlit

Streamlit companion for the corrected MS AI Assignment 1. The app reads fresh precomputed seed-42 results and never performs hyperparameter tuning in the cloud.

## Streamlit Cloud configuration
- Repository: `mehwishrani081-png/MS-AI-Fraud-Detection-Streamlit`
- Branch: `main`
- Main file: `app.py`
- Runtime: `python-3.13.5`

## Required artifact layout
The complete local package contains:
```
outputs/
  results.json
  figures/
  tables/
  models/final_supervised_model.joblib
  sample_test_transactions.csv
```
The connected GitHub repository currently contains the fresh text artifacts/tables. The app fails gracefully to actual precomputed evaluated transactions if the binary model/sample artifacts are absent. The downloadable submission contains the complete model/sample/figure package and the upload-capable app.

## Local test
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Exact push commands for the complete artifact package
Copy the downloadable `streamlit_app/outputs/` folder into this repository, then:
```bash
git add app.py requirements.txt runtime.txt .gitignore README.md outputs/
git commit -m "Deploy corrected fresh MS AI fraud-detection artifacts"
git push origin main
```

Do **not** commit `creditcard.csv`. Streamlit Community Cloud automatically redeploys the tracked `main` branch. Confirm the new build under **Manage app → Logs**; use **Reboot app** only if the new commit is visible but the process remains stale.
