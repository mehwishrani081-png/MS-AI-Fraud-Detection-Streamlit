from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn
import shap

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
MODEL_PATH = OUT / "models" / "final_supervised_model.joblib"
RESULTS_PATH = OUT / "results.json"
SAMPLE_PATH = OUT / "sample_test_transactions.csv"
EXPECTED_SKLEARN = "1.8.0"

st.set_page_config(page_title="MS AI Fraud Detection", page_icon="🛡️", layout="wide")

@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model) if model is not None else None

R = load_json(RESULTS_PATH)
if R is None:
    st.error("Required artifact outputs/results.json is missing. Redeploy after restoring the outputs folder.")
    st.stop()

if sklearn.__version__ != EXPECTED_SKLEARN:
    st.warning(f"Model compatibility warning: app expects scikit-learn {EXPECTED_SKLEARN}, but runtime has {sklearn.__version__}.")

st.title("🛡️ Detecting Fraud in Imbalanced, Evolving Data")
st.caption("MS Artificial Intelligence — Track A: ULB Credit Card Fraud Detection | Fresh seed-42 executed results")

pages = ["Overview","Model Comparison","Evaluation","Anomaly Detection","Drift","Explainability","Recommendation","Try a transaction"]
page = st.sidebar.radio("Navigation", pages)
st.sidebar.caption("All metrics are loaded from precomputed fresh-run artifacts; the app does not re-tune models.")

def show_img(name, caption=None):
    p = FIG / name
    if p.exists(): st.image(str(p), caption=caption, use_container_width=True)
    else: st.info(f"Figure artifact missing: {name}")

def show_table(name):
    d = load_csv(TAB / name)
    if d is None: st.info(f"Table artifact missing: {name}")
    else: st.dataframe(d, use_container_width=True, hide_index=True)

if page == "Overview":
    ds=R["dataset"]; raw=R["class_counts_raw"]; clean=R["class_counts_clean"]
    c=st.columns(5)
    c[0].metric("Raw transactions", f"{ds['raw_shape'][0]:,}")
    c[1].metric("Raw frauds", f"{raw['1']:,}")
    c[2].metric("Fraud prevalence", f"{100*raw['1']/ds['raw_shape'][0]:.4f}%")
    c[3].metric("Exact duplicates", f"{R['duplicates_total']:,}")
    c[4].metric("Clean rows", f"{R['clean_shape'][0]:,}")
    st.markdown(f"**Source:** {ds['source']}  \n**SHA256:** `{ds['sha256']}`  \n**Main-data sampling:** {ds['sampling_main']}  \n**Clean imbalance ratio:** {R['imbalance_ratio_clean']:.2f}:1")
    show_img("class_distribution.png","Class distribution")
    st.subheader("Random and temporal protocols")
    st.json({"random_split":R["random_split"],"temporal_split":R["temporal_split"],"tie_handling":R["temporal_tie_handling"]})

elif page == "Model Comparison":
    st.header("Supervised model and imbalance-strategy comparison")
    show_table("supervised_tuning.csv")
    strategies=load_csv(TAB/"strategies.csv")
    if strategies is not None:
        model_col=next((c for c in strategies.columns if c.lower()=="model"),None)
        if model_col:
            chosen=st.multiselect("Filter models", sorted(strategies[model_col].unique()), default=sorted(strategies[model_col].unique()))
            strategies=strategies[strategies[model_col].isin(chosen)]
        st.dataframe(strategies,use_container_width=True,hide_index=True)
    st.success(f"Selection rule: {R['selection_rule']} Final model: **{R['final_model']}**.")
    st.markdown(f"Correct-pipeline CV PR-AUC vs deliberate pre-CV SMOTE leakage: **{R['leakage_demo']['proper_mean']:.4f} vs {R['leakage_demo']['leaky_smote_before_cv_mean']:.4f}**.")

elif page == "Evaluation":
    st.header("Rare-event evaluation")
    show_table("all_final_metrics.csv")
    show_img("roc_top3.png","ROC curves — top three models")
    show_img("pr_top3.png","Precision–Recall curves — top three models with no-skill reference")
    st.subheader("Cost-sensitive decisions")
    show_table("costs.csv")
    m=R["mcnemar"]
    st.markdown(f"**McNemar:** table `{m['table']}`, b={m['b']}, c={m['c']}, disagreements={m['disagreements']}, exact p={m['exact_p']:.4f}, continuity-corrected p={m['cc_p']:.4f}.")
    st.markdown(f"**Wilcoxon 5-fold PR-AUC check:** statistic={R['wilcoxon']['stat']:.4f}, p={R['wilcoxon']['p']:.4f}.")
    st.json(R["bootstrap_ci"])

elif page == "Anomaly Detection":
    st.header("Unsupervised and semi-supervised anomaly detection")
    show_table("anomaly.csv")
    st.success(f"Best anomaly configuration: **{R['best_anomaly']}**")
    st.subheader("Repeated label-budget experiment")
    show_table("label_budget_summary.csv")
    show_img("label_budget.png","PR-AUC versus labelled training budget (mean ± SD)")
    st.markdown(f"Supervision first exceeded the best anomaly reference at **{R['label_budget_first_win']}%** labelled data; the 100% point matches the main final-model PR-AUC: **{R['label_budget_100_matches_main']}**.")

elif page == "Drift":
    st.header("Temporal drift")
    show_table("drift.csv")
    show_img("drift.png","Supervised and anomaly-detector performance across temporal windows")
    st.subheader("Feature drift (KS tests)")
    show_table("ks_drift.csv")
    st.markdown("Windows with fewer than 10 frauds are explicitly treated as statistically unstable in the report. Monitoring includes data quality, fraud prevalence, feature drift, delayed-label performance, threshold review, retraining triggers, analyst feedback and rollback.")

elif page == "Explainability":
    st.header("Global and local explanations")
    c1,c2=st.columns(2)
    with c1: show_img("native_importance.png","Impurity-based importance")
    with c2: show_img("permutation_importance.png","Permutation importance on validation PR-AUC")
    show_img("shap_summary.png","SHAP summary — final Random Forest")
    st.subheader("Six required local cases")
    for case in R["shap_cases"]:
        with st.expander(f"Case {case['case']} — {case['type']} — score {case['score']:.4f}"):
            st.write(case["line1"])
            st.write(case["line2"])
            show_img(f"shap_waterfall_{case['case']}_{case['type']}.png")
    st.subheader("Classifier–anomaly agreement")
    st.json(R["agreement"])

elif page == "Recommendation":
    e=R["recommendation_evidence"]
    st.header("Evidence-based deployment recommendation")
    st.success(f"Primary rule: {e['final_model']} — {e['final_strategy']}")
    c=st.columns(4)
    c[0].metric("PR-AUC",f"{e['pr_auc']:.4f}"); c[1].metric("Precision",f"{e['precision']:.4f}"); c[2].metric("Recall",f"{e['recall']:.4f}"); c[3].metric("Cost saving",f"{e['saving_vs_nothing_pct']:.2f}%")
    st.markdown(f"The final threshold is **{e['threshold']:.4f}** and test cost is **{e['cost']:.2f}**. Semi-supervised LOF remains a secondary analyst-review signal (PR-AUC **{e['best_anomaly_pr_auc']:.4f}**). The tested rank-average hybrid is **not** recommended because its cost (**{e['hybrid_cost']:.2f}**) exceeds the supervised cost and its recall (**{e['hybrid_recall']:.4f}**) is lower.")
    st.warning("Limitations: few frauds in the test set, PCA-transformed V1–V28, no customer/merchant identifiers, one public dataset, and a short approximately two-day observation window.")

elif page == "Try a transaction":
    st.header("Try a transaction")
    st.caption("This cloud-safe panel uses actual precomputed test cases and SHAP outputs from the fresh run; it does not fabricate a new prediction.")
    cases=R["shap_cases"]
    labels=[f"Case {c['case']} — {c['type']} — source_index {c['source_index']}" for c in cases]
    chosen=st.selectbox("Pick an actual evaluated test transaction", range(len(cases)), format_func=lambda i: labels[i])
    case=cases[chosen]
    prob=float(case["score"]); thr=float(R["final_operational_threshold"]); decision="FRAUD ALERT" if prob>=thr else "No alert"
    c=st.columns(4)
    c[0].metric("True class",case["true"])
    c[1].metric("Fraud probability",f"{prob:.4f}")
    c[2].metric("Cost-optimal threshold",f"{thr:.4f}")
    c[3].metric("Decision",decision)
    st.subheader("Local SHAP evidence")
    st.dataframe(pd.DataFrame(case["top"]),use_container_width=True,hide_index=True)
    st.write(case["line1"])
    st.write(case["line2"])
    show_img(f"shap_waterfall_{case['case']}_{case['type']}.png")
    if case.get("fn_investigation"):
        st.subheader("False-negative investigation")
        st.json(case["fn_investigation"])

st.divider()
st.caption("Academic demonstration only; not a production banking decision system.")
