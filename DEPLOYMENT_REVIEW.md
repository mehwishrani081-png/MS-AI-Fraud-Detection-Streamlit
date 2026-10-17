# Streamlit deployment review

The updated fraud-detection Streamlit package from this assignment has **not yet been fully synchronized** with this GitHub branch.

## Required contents before deployment
- `app.py`
- `requirements.txt`
- `results/results.json`
- `results/models/best_supervised.joblib`
- `results/figures/`
- `results/tables/`

The currently deployed/committed app on `main` expects a **different** results layout (`outputs/`) and should not be confused with the newer standalone package. Do not replace the main app with the new app without also uploading its model and results artifacts.

## Deploy to Streamlit Community Cloud
1. Upload all contents of `ADM_Streamlit_Deploy_Ready.zip` to a dedicated branch or repository.
2. Ensure the `results/` directory is present and the large model file is tracked.
3. Open https://share.streamlit.io/, connect this GitHub repository, select `app.py` as entrypoint and publish.
4. Verify all pages, model prediction, and CSV uploads before sharing the live URL.

This note records a preparation step, not proof of a successful live deployment.
