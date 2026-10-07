from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn

BASE = Path(__file__).resolve().parent

def resolve_outputs():
    direct = BASE / "outputs"
    if (direct / "results.json").exists():
        return direct
    chunks = sorted((BASE / "artifact_chunks").glob("core_*.part"))
    if chunks:
        import base64, io, tempfile, zipfile
        target = Path(tempfile.gettempdir()) / "fraud_streamlit_artifacts"
        marker = target / "outputs" / "results.json"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            payload = base64.b64decode("".join(x.read_text().strip() for x in chunks))
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                zf.extractall(target)
        if marker.exists():
            return target / "outputs"
    return direct

OUT = resolve_outputs()
FIG = OUT / "figures"
TAB = OUT / "tables"
MODEL_PATH = OUT / "models" / "final_supervised_model.joblib"
RESULTS_PATH = OUT / "results.json"
SAMPLE_PATH = OUT / "sample_test_transactions.csv"
EXPECTED_SKLEARN = "1.8.0"

st.set_page_config(page_title="MS AI Fraud Detection", page_icon="🛡️", layout="wide")

@st.cache_data
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_explainer(model):
    if model is None:
        return None
    import shap
    return shap.TreeExplainer(model)

R = load_json(RESULTS_PATH)
if R is None:
    st.error("Fresh precomputed artifact results.json is missing.")
    st.stop()

if sklearn.__version__ != EXPECTED_SKLEARN:
    st.warning(f"Model compatibility warning: expected scikit-learn {EXPECTED_SKLEARN}; found {sklearn.__version__}.")

st.title("🛡️ Detecting Fraud in Imbalanced, Evolving Data")
st.caption("MS Artificial Intelligence — Track A | fresh seed-42 executed artifacts")
pages=["Overview","Model Comparison","Evaluation","Anomaly Detection","Drift","Explainability","Recommendation","Try a transaction"]
page=st.sidebar.radio("Navigation",pages)
st.sidebar.caption("The cloud app loads precomputed artifacts; it never re-tunes models.")

def show_img(name,caption=None):
    p=FIG/name
    if p.exists():
        st.image(str(p),caption=caption,use_container_width=True)
        return
    fallback={
        "drift.png":("drift.csv",["supervised_precision","supervised_recall","anomaly_precision","anomaly_recall"],"window"),
        "label_budget.png":("label_budget_summary.csv",["mean"],"budget_pct"),
    }
    spec=fallback.get(name)
    if spec:
        d=load_csv(TAB/spec[0])
        if d is not None: st.line_chart(d.set_index(spec[2])[spec[1]])
    else:
        st.caption(f"Precomputed figure {name} is included in the downloadable submission package.")

def show_table(name):
    d=load_csv(TAB/name)
    if d is None: st.info(f"Missing table artifact: {name}")
    else: st.dataframe(d,use_container_width=True,hide_index=True)

if page=="Overview":
    ds=R["dataset"]; raw=R["class_counts_raw"]
    c=st.columns(5)
    c[0].metric("Raw transactions",f"{ds['raw_shape'][0]:,}")
    c[1].metric("Raw frauds",f"{raw['1']:,}")
    c[2].metric("Fraud prevalence",f"{100*raw['1']/ds['raw_shape'][0]:.4f}%")
    c[3].metric("Exact duplicates",f"{R['duplicates_total']:,}")
    c[4].metric("Clean rows",f"{R['clean_shape'][0]:,}")
    st.markdown(f"**Source:** {ds['source']}  \n**SHA256:** {ds['sha256']}  \n**Main-data sampling:** {ds['sampling_main']}  \n**Clean imbalance ratio:** {R['imbalance_ratio_clean']:.2f}:1")
    show_img("class_distribution.png","Class distribution")
    st.json({"random_split":R["random_split"],"temporal_split":R["temporal_split"],"tie_handling":R["temporal_tie_handling"]})

elif page=="Model Comparison":
    st.header("Supervised model × strategy comparison")
    show_table("supervised_tuning.csv")
    d=load_csv(TAB/"strategies.csv")
    if d is not None:
        chosen=st.multiselect("Filter models",sorted(d.model.unique()),default=sorted(d.model.unique()))
        st.dataframe(d[d.model.isin(chosen)],use_container_width=True,hide_index=True)
    st.success(f"Selection rule: {R['selection_rule']} Final model: **{R['final_model']}**.")
    st.markdown(f"Correct-pipeline CV PR-AUC vs deliberate pre-CV SMOTE leakage: **{R['leakage_demo']['proper_mean']:.4f} vs {R['leakage_demo']['leaky_smote_before_cv_mean']:.4f}**.")

elif page=="Evaluation":
    st.header("Rare-event evaluation")
    show_table("all_final_metrics.csv")
    show_img("roc_top3.png","ROC — top three")
    show_img("pr_top3.png","Precision–Recall — top three")
    st.subheader("Cost-sensitive decisions"); show_table("costs.csv")
    m=R["mcnemar"]
    st.markdown(f"**McNemar:** table {m['table']}, b={m['b']}, c={m['c']}, disagreements={m['disagreements']}, exact p={m['exact_p']:.4f}, continuity-corrected p={m['cc_p']:.4f}.")
    st.markdown(f"**Wilcoxon:** statistic={R['wilcoxon']['stat']:.4f}, p={R['wilcoxon']['p']:.4f}.")
    st.json(R["bootstrap_ci"])

elif page=="Anomaly Detection":
    st.header("Unsupervised and semi-supervised anomaly detection")
    show_table("anomaly.csv")
    st.success(f"Best anomaly configuration: **{R['best_anomaly']}**")
    show_table("label_budget_summary.csv"); show_img("label_budget.png","Label-budget PR-AUC")
    st.markdown(f"Supervision first beats the best anomaly reference at **{R['label_budget_first_win']}%** labels; 100% matches main result: **{R['label_budget_100_matches_main']}**.")

elif page=="Drift":
    st.header("Temporal drift")
    show_table("drift.csv"); show_img("drift.png","Temporal precision/recall")
    st.subheader("KS feature-drift tests"); show_table("ks_drift.csv")
    st.markdown("Windows with fewer than 10 frauds are explicitly treated as unstable; monitoring includes data quality, prevalence, KS/PSI drift, delayed-label performance, threshold review, retraining triggers, analyst feedback and rollback.")

elif page=="Explainability":
    st.header("Global and local explanations")
    c1,c2=st.columns(2)
    with c1: show_img("native_importance.png","Impurity importance")
    with c2: show_img("permutation_importance.png","Permutation importance")
    show_img("shap_summary.png","SHAP summary")
    for case in R["shap_cases"]:
        with st.expander(f"Case {case['case']} — {case['type']} — score {case['score']:.4f}"):
            st.write(case["line1"]); st.write(case["line2"])
            show_img(f"shap_waterfall_{case['case']}_{case['type']}.png")
    st.subheader("Classifier–anomaly agreement"); st.json(R["agreement"])

elif page=="Recommendation":
    e=R["recommendation_evidence"]
    st.header("Evidence-based recommendation")
    st.success(f"Primary rule: {e['final_model']} — {e['final_strategy']}")
    c=st.columns(4)
    c[0].metric("PR-AUC",f"{e['pr_auc']:.4f}"); c[1].metric("Precision",f"{e['precision']:.4f}"); c[2].metric("Recall",f"{e['recall']:.4f}"); c[3].metric("Cost saving",f"{e['saving_vs_nothing_pct']:.2f}%")
    st.markdown(f"Threshold **{e['threshold']:.4f}**, test cost **{e['cost']:.2f}**. Semi-supervised LOF PR-AUC **{e['best_anomaly_pr_auc']:.4f}** remains a secondary review signal. Tested RF+LOF fusion is rejected because cost **{e['hybrid_cost']:.2f}** is higher and recall **{e['hybrid_recall']:.4f}** is lower.")
    st.warning("Limitations: few frauds, PCA-transformed features, no customer/merchant IDs, one dataset, short historical window.")

elif page=="Try a transaction":
    st.header("Try a transaction")
    model=load_model(MODEL_PATH); sample=load_csv(SAMPLE_PATH)
    features=["Time"]+[f"V{i}" for i in range(1,29)]+["Amount"]
    if model is None or sample is None:
        st.warning("The connected GitHub deployment is missing the binary model/sample artifacts. Showing an actual precomputed test-case fallback. The downloadable submission includes the full model and upload-capable app.")
        cases=R["shap_cases"]
        labels=[f"Case {c['case']} — {c['type']} — source_index {c['source_index']}" for c in cases]
        chosen=st.selectbox("Pick an actual evaluated test transaction",range(len(cases)),format_func=lambda k:labels[k])
        case=cases[chosen]
        prob=float(case["score"]); thr=float(R["final_operational_threshold"]); decision="FRAUD ALERT" if prob>=thr else "No alert"
        c=st.columns(4)
        c[0].metric("True class",case["true"]); c[1].metric("Fraud probability",f"{prob:.4f}"); c[2].metric("Cost-optimal threshold",f"{thr:.4f}"); c[3].metric("Decision",decision)
        st.subheader("Actual precomputed SHAP evidence")
        st.dataframe(pd.DataFrame(case["top"]),use_container_width=True,hide_index=True)
        st.write(case["line1"]); st.write(case["line2"])
        if case.get("fn_investigation"): st.json(case["fn_investigation"])
    else:
        mode=st.radio("Input",["Pick a saved test transaction","Upload CSV"])
        row=None
        if mode=="Pick a saved test transaction":
            idx=st.selectbox("Sample row",list(range(len(sample))),format_func=lambda k:f"Row {k} | true Class={int(sample.iloc[k]['Class'])} | source_index={sample.iloc[k].get('source_index','n/a')}")
            row=sample.iloc[[idx]][features].copy()
        else:
            up=st.file_uploader("Upload CSV with the 30 required feature columns",type=["csv"])
            if up is not None:
                u=pd.read_csv(up); missing=[c for c in features if c not in u.columns]
                if missing: st.error("Missing columns: "+", ".join(missing))
                else: row=u[features].iloc[[0]].copy()
        if row is not None:
            prob=float(model.predict_proba(row)[0,1]); thr=float(R["final_operational_threshold"]); decision="FRAUD ALERT" if prob>=thr else "No alert"
            c=st.columns(3); c[0].metric("Fraud probability",f"{prob:.4f}"); c[1].metric("Cost-optimal threshold",f"{thr:.4f}"); c[2].metric("Decision",decision)
            st.dataframe(row,use_container_width=True,hide_index=True)
            try:
                ex=load_explainer(model); sv=ex.shap_values(row); sv=np.asarray(sv[1] if isinstance(sv,list) else sv); sv=sv[:,:,1] if sv.ndim==3 else sv
                vals=sv[0]; order=np.argsort(np.abs(vals))[::-1][:8]
                contrib=pd.DataFrame({"feature":[features[k] for k in order],"value":[float(row.iloc[0,k]) for k in order],"shap":[float(vals[k]) for k in order],"direction":["toward fraud" if vals[k]>0 else "toward legitimate" for k in order]})
                st.subheader("Local SHAP explanation"); st.dataframe(contrib,use_container_width=True,hide_index=True)
                st.caption("V1–V28 are PCA components; SHAP direction is model contribution, not an original business attribute.")
            except Exception as exc:
                st.warning(f"Prediction succeeded; SHAP failed gracefully: {exc}")

st.divider()
st.caption("Academic demonstration only; not a production banking decision system.")
