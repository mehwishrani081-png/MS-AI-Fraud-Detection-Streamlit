# MS-AI-Fraud-Detection-Streamlit

Streamlit companion for the corrected MS AI Assignment 1. The app loads **precomputed fresh-run artifacts** from `outputs/` and does not re-tune models in the cloud.

## Streamlit Cloud
- Repository: `mehwishrani081-png/MS-AI-Fraud-Detection-Streamlit`
- Branch: `main`
- Main file: `app.py`
- Python runtime: `python-3.13.5`

## Local test
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Git push commands
```bash
git add app.py requirements.txt runtime.txt .gitignore README.md outputs/
git commit -m "Update Streamlit app with corrected fresh assignment artifacts"
git push origin main
```

## Redeploy check
Streamlit Community Cloud auto-redeploys the tracked `main` branch. Open **Manage app → Logs** to confirm the new commit was built. Use **Reboot app** only if the cloud process remains stale after the new commit is visible.

`creditcard.csv` is intentionally not committed. The demo uses a 300-row test sample and the compressed final Random Forest model.
