import streamlit as st
import pandas as pd

st.set_page_config(page_title="MS AI Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Detecting Fraud in Imbalanced, Evolving Data")
st.caption("Track A — ULB Credit Card Fraud Detection | Interactive dashboard based only on executed assignment results")

RAW_SHAPE=(284807,31)
CLEAN_SHAPE=(283726,31)
CLASS_RAW={"Legitimate":284315,"Fraud":492}
CLASS_CLEAN={"Legitimate":283253,"Fraud":473}
MISSING=0
INFINITE=0
CONSTANT_FEATURES=[]
DUPLICATES=1081
DUP_REMOVED={"Legitimate":1062,"Fraud":19}
IMBALANCE_RAW=577.8760162601626
IMBALANCE_CLEAN=598.8435517970402

TOP_MI=[
("V14",0.008291711390899459),("V17",0.008027960260461708),("V10",0.007768695180292995),
("V12",0.0076502945397433075),("V11",0.006778206407573761),("V16",0.006200927427517988),
("V3",0.005265411548870058),("V4",0.005122769564628649),("V9",0.004460101944756323),
("V7",0.004427225564869297)
]

SUPERVISED=[
{"model":"Random Forest","cv_pr_auc":0.8112303299144731,"val_pr_auc":0.8146383878536064,"test_pr_auc":0.7779729195947849,"test_roc_auc":0.9265133021817594,"test_precision":0.8947368421052632,"test_recall":0.7183098591549296,"test_f1":0.796875,"test_mcc":0.8013950322763703},
{"model":"k-NN","cv_pr_auc":0.7896,"val_pr_auc":0.7507543270521965,"test_pr_auc":0.7373424706440187,"test_roc_auc":0.8800201415610969,"test_precision":0.9107142857142856,"test_recall":0.7183098591549296,"test_f1":0.8031496062992126,"test_mcc":0.8085356691260973},
{"model":"Logistic Regression","cv_pr_auc":0.758262919492872,"val_pr_auc":0.690628684323855,"test_pr_auc":0.7226767935148797,"test_roc_auc":0.965009175747386,"test_precision":0.8627450980392157,"test_recall":0.6197183098591549,"test_f1":0.7213114754098361,"test_mcc":0.7308373759015526},
{"model":"Decision Tree","cv_pr_auc":0.6798814451280998,"val_pr_auc":0.6870161862922374,"test_pr_auc":0.6941982959332735,"test_roc_auc":0.8855385182493947,"test_precision":0.9272727272727272,"test_recall":0.7183098591549296,"test_f1":0.8095238095238095,"test_mcc":0.8158700219342182},
{"model":"LightGBM","cv_pr_auc":0.7639409994724353,"val_pr_auc":0.1327797207364746,"test_pr_auc":0.1846341881714447,"test_roc_auc":0.6584758977514116,"test_precision":0.25,"test_recall":0.5352112676056338,"test_f1":0.3408071748878923,"test_mcc":0.3643043233556565},
{"model":"Gaussian Naive Bayes","cv_pr_auc":0.0792629787116738,"val_pr_auc":0.0772498932456933,"test_pr_auc":0.0796130897243471,"test_roc_auc":0.9594719039145436,"test_precision":0.0576540755467196,"test_recall":0.8169014084507042,"test_f1":0.1077065923862581,"test_mcc":0.2134542510308789}
]

IMBALANCE=[
{"model":"Random Forest","strategy":"No treatment","val_pr_auc":0.8079084738479377,"test_precision":0.9107142857142857,"test_recall":0.7183098591549296,"test_f1":0.8031496062992126,"test_mcc":0.8085356691260973,"test_pr_auc":0.7624428101476419},
{"model":"Random Forest","strategy":"Class weights","val_pr_auc":0.7932784033406267,"test_precision":0.8392857142857143,"test_recall":0.6619718309859155,"test_f1":0.7401574803149606,"test_mcc":0.7450047290622487,"test_pr_auc":0.7659351888340256},
{"model":"Random Forest","strategy":"Random undersampling","val_pr_auc":0.6445177040986647,"test_precision":0.06521739130434782,"test_recall":0.8450704225352113,"test_f1":0.12108980827447023,"test_mcc":0.23146343762381175,"test_pr_auc":0.6413863248606283},
{"model":"Random Forest","strategy":"SMOTE","val_pr_auc":0.78187722818329,"test_precision":0.803030303030303,"test_recall":0.7464788732394366,"test_f1":0.7737226277372263,"test_mcc":0.7738755554859057,"test_pr_auc":0.7649639320188151},
{"model":"Random Forest","strategy":"Threshold moving","val_pr_auc":0.78187722818329,"test_precision":0.8253968253968254,"test_recall":0.7323943661971831,"test_f1":0.7761194029850746,"test_mcc":0.7771582553839695,"test_pr_auc":0.7649639320188151},
{"model":"Logistic Regression","strategy":"No treatment","val_pr_auc":0.690628684323855,"test_precision":0.8627450980392157,"test_recall":0.6197183098591549,"test_f1":0.7213114754098361,"test_mcc":0.7308373759015526,"test_pr_auc":0.7226767935148797},
{"model":"Logistic Regression","strategy":"Class weights","val_pr_auc":0.6904063257757143,"test_precision":0.05526315789473684,"test_recall":0.8873239436619719,"test_f1":0.10404624277456648,"test_mcc":0.2178738288789055,"test_pr_auc":0.6764729698507778},
{"model":"Logistic Regression","strategy":"Random undersampling","val_pr_auc":0.4732194074667663,"test_precision":0.09904153354632587,"test_recall":0.8732394366197183,"test_f1":0.17790530846484937,"test_mcc":0.29152648191974645,"test_pr_auc":0.554540417856388},
{"model":"Logistic Regression","strategy":"SMOTE","val_pr_auc":0.6981643100042424,"test_precision":0.13606911447084233,"test_recall":0.8873239436619719,"test_f1":0.23595505617977527,"test_mcc":0.3453831662987147,"test_pr_auc":0.6956474124371579},
{"model":"Logistic Regression","strategy":"Threshold moving","val_pr_auc":0.6981643100042424,"test_precision":0.8333333333333334,"test_recall":0.7746478873239436,"test_f1":0.8029197080291971,"test_mcc":0.8031392010154195,"test_pr_auc":0.6956474124371579}
]

COST=[
{"model":"Random Forest","validation_cost_min":887.70,"selected_threshold":0.18032421479229852,"test_total_cost":3481.74,"test_precision":0.5327102803738317,"test_recall":0.8028169014084507,"test_f1":0.6404494382022472},
{"model":"Logistic Regression","validation_cost_min":875.34,"selected_threshold":0.9356936036810534,"test_total_cost":3470.74,"test_precision":0.5471698113207547,"test_recall":0.8169014084507042,"test_f1":0.655367231638418}
]

ANOMALY=[
{"detector":"Local Outlier Factor","setting":"Semi-supervised","test_pr_auc":0.509844888866131,"test_roc_auc":0.9599747799544395,"test_precision_at_k":0.6338028169014085},
{"detector":"One-Class SVM","setting":"Semi-supervised","test_pr_auc":0.1825551134580359,"test_roc_auc":0.9423381514846942,"test_precision_at_k":0.2112676056338028},
{"detector":"Isolation Forest","setting":"Unsupervised","test_pr_auc":0.14662349411006317,"test_roc_auc":0.9564340950618037,"test_precision_at_k":0.23943661971830985},
{"detector":"Isolation Forest","setting":"Semi-supervised","test_pr_auc":0.12979927471707747,"test_roc_auc":0.9560041476499744,"test_precision_at_k":0.2112676056338028},
{"detector":"One-Class SVM","setting":"Unsupervised","test_pr_auc":0.0786678522075928,"test_roc_auc":0.9374913811621376,"test_precision_at_k":0.14084507042253522},
{"detector":"Local Outlier Factor","setting":"Unsupervised","test_pr_auc":0.02897521097333896,"test_roc_auc":0.8857165304006301,"test_precision_at_k":0.08450704225352113}
]

LABEL_BUDGET=[
{"label_budget_pct":1,"training_n":1986,"fraud_n":3,"test_pr_auc":0.6155005858146971},
{"label_budget_pct":5,"training_n":9930,"fraud_n":17,"test_pr_auc":0.7140815797499446},
{"label_budget_pct":10,"training_n":19860,"fraud_n":33,"test_pr_auc":0.7341859909204282},
{"label_budget_pct":25,"training_n":49652,"fraud_n":83,"test_pr_auc":0.7466087216642272},
{"label_budget_pct":100,"training_n":198608,"fraud_n":331,"test_pr_auc":0.796949185468741}
]

TEMPORAL=[
{"window":1,"fraud_n":25,"fraud_rate":0.002349624060150376,"supervised_precision":1.0,"supervised_recall":0.76,"anomaly_precision":0.8333333333333334,"anomaly_recall":0.6},
{"window":2,"fraud_n":14,"fraud_rate":0.0013157894736842105,"supervised_precision":0.9166666666666666,"supervised_recall":0.7857142857142857,"anomaly_precision":0.42857142857142855,"anomaly_recall":0.42857142857142855},
{"window":3,"fraud_n":3,"fraud_rate":0.00028195488721804513,"supervised_precision":0.5,"supervised_recall":0.6666666666666666,"anomaly_precision":0.6666666666666666,"anomaly_recall":0.6666666666666666},
{"window":4,"fraud_n":10,"fraud_rate":0.0009399379640943697,"supervised_precision":0.5555555555555556,"supervised_recall":0.5,"anomaly_precision":0.5714285714285714,"anomaly_recall":0.4}
]

NATIVE=[
("V17",0.21884725517090428),("V14",0.20281796374974848),("V12",0.09091042160248884),
("V10",0.08141088653960034),("V16",0.062480461626742284),("V9",0.04275974254527747),
("V4",0.02835948308720028),("V3",0.026562236023892592),("V20",0.020051629433121287),("V26",0.019359460808570747)
]
PERM=[
("V14",0.20002944697726746),("V17",0.06322903128438362),("V12",0.026718028689078948),
("V16",0.017886141825151136),("V26",0.015649522017099755),("V27",0.014145648448358342),
("V4",0.013857963076822041),("V11",0.01242265612127206),("V20",0.009642082828684395),("V10",0.009631101017257023)
]

SHAP_CASES=[
{"case_type":"TP","score":0.9833333333333332,"top_features":"V14, V17, V12, V10, V20"},
{"case_type":"TP","score":0.7608333333333334,"top_features":"V14, V12, V4, V17, V9"},
{"case_type":"TP","score":0.925,"top_features":"V14, V17, V10, V16, V12"},
{"case_type":"FP","score":0.862142857142857,"top_features":"V17, V14, V10, V12, V21"},
{"case_type":"FP","score":0.5638888888888889,"top_features":"V14, V12, V17, V10, V16"},
{"case_type":"FN","score":0.0,"top_features":"V14, V10, V3, V17, Amount"}
]

RANDOM_SPLIT=pd.DataFrame([
{"split":"Train","n":198608,"fraud":331,"rate":0.0016665995327479256},
{"split":"Validation","n":42559,"fraud":71,"rate":0.0016682722808336662},
{"split":"Test","n":42559,"fraud":71,"rate":0.0016682722808336662}
])

TEMPORAL_SPLIT=pd.DataFrame([
{"split":"Train","n":198608,"fraud":366,"rate":0.0018428260694433255,"time_min":0.0,"time_max":132906.0},
{"split":"Validation","n":42559,"fraud":55,"rate":0.0012923235978288964,"time_min":132906.0,"time_max":151320.0},
{"split":"Test","n":42559,"fraud":52,"rate":0.001221833219765502,"time_min":151320.0,"time_max":172792.0}
])

page=st.sidebar.radio("Navigation",[
"Overview","Data Understanding","Supervised Models","Imbalance Strategies",
"Rare-Event Evaluation","Anomaly Detection","Label Budget","Temporal Drift",
"Explainability","Leakage Audit"
])
st.sidebar.info("All displayed values are actual outputs from the executed notebook. No synthetic metrics are generated.")

def metrics(items):
    cols=st.columns(len(items))
    for c,(k,v) in zip(cols,items):
        c.metric(k,v)

if page=="Overview":
    metrics([
        ("Raw transactions",f"{RAW_SHAPE[0]:,}"),
        ("Columns",RAW_SHAPE[1]),
        ("Fraud cases",f"{CLASS_RAW['Fraud']:,}"),
        ("Fraud prevalence",f"{100*CLASS_RAW['Fraud']/RAW_SHAPE[0]:.4f}%"),
        ("Exact duplicates",f"{DUPLICATES:,}")
    ])
    st.subheader("Class distribution after exact-duplicate handling")
    st.bar_chart(pd.DataFrame({"Class":CLASS_CLEAN.keys(),"Count":CLASS_CLEAN.values()}).set_index("Class"))
    st.write(f"Cleaned dataset: **{CLEAN_SHAPE[0]:,} rows × {CLEAN_SHAPE[1]} columns**.")
    st.write(f"Cleaned imbalance ratio: **{IMBALANCE_CLEAN:.2f}:1** legitimate-to-fraud.")

elif page=="Data Understanding":
    st.header("Task 1.1 — Data Understanding and Preprocessing")
    metrics([("Missing values",MISSING),("Infinite values",INFINITE),("Constant features",0),("|r| > 0.95 pairs",0)])
    st.subheader("Duplicate analysis")
    st.write(f"Exact duplicate copies identified: **{DUPLICATES:,}**.")
    st.dataframe(pd.DataFrame({"Class":DUP_REMOVED.keys(),"Removed duplicate copies":DUP_REMOVED.values()}),hide_index=True,use_container_width=True)
    st.subheader("Correlation analysis")
    st.success("No predictor pair exceeded |r| > 0.95.")
    st.subheader("Ten most informative features")
    mi=pd.DataFrame(TOP_MI,columns=["Feature","Mutual information"])
    st.dataframe(mi,hide_index=True,use_container_width=True)
    st.bar_chart(mi.set_index("Feature"))
    st.subheader("Random stratified 70/15/15 split")
    st.dataframe(RANDOM_SPLIT,hide_index=True,use_container_width=True)
    st.subheader("Time-ordered split")
    st.dataframe(TEMPORAL_SPLIT,hide_index=True,use_container_width=True)

elif page=="Supervised Models":
    st.header("Task 1.2 — Supervised Classification")
    df=pd.DataFrame(SUPERVISED).sort_values("val_pr_auc",ascending=False)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.success("Best supervised model by validation PR-AUC: **Random Forest**.")
    st.subheader("Majority-class baseline")
    metrics([("Accuracy","99.833%"),("Precision","0.000"),("Recall","0.000"),("F1","0.000")])
    st.warning("The baseline shows why accuracy is misleading: nearly 99.8% accuracy while detecting zero frauds.")

elif page=="Imbalance Strategies":
    st.header("Five Required Imbalance Strategies")
    df=pd.DataFrame(IMBALANCE)
    model=st.selectbox("Model",sorted(df["model"].unique()))
    st.dataframe(df[df["model"]==model],hide_index=True,use_container_width=True)
    st.caption("Compared strategies: no treatment, class weights, random undersampling, SMOTE, and validation-set threshold moving.")

elif page=="Rare-Event Evaluation":
    st.header("Task 1.3 — Rare-Event Evaluation")
    st.subheader("Cost-sensitive threshold selection")
    st.dataframe(pd.DataFrame(COST),hide_index=True,use_container_width=True)
    st.subheader("McNemar test: Random Forest vs Logistic Regression")
    metrics([("b",388),("c",11),("Statistic","354.326"),("p-value","4.84 × 10⁻79")])
    st.success("Difference is statistically significant at α = 0.05.")
    st.info("PR-AUC is emphasized because it is more informative than accuracy and often more revealing than ROC-AUC under extreme class imbalance.")

elif page=="Anomaly Detection":
    st.header("Task 1.4 — Anomaly Detection")
    df=pd.DataFrame(ANOMALY).sort_values("test_pr_auc",ascending=False)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.success("Best anomaly configuration: **Local Outlier Factor — Semi-supervised**, test PR-AUC = **0.5098**.")

elif page=="Label Budget":
    st.header("Label-Budget Experiment")
    df=pd.DataFrame(LABEL_BUDGET)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.line_chart(df.set_index("label_budget_pct")[["test_pr_auc"]])
    st.ax=None
    st.info("Supervised learning already exceeded the best anomaly-detector reference PR-AUC (0.5098) at the 1% label budget in this executed experiment.")

elif page=="Temporal Drift":
    st.header("Task 1.5 — Temporal Drift")
    df=pd.DataFrame(TEMPORAL)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.line_chart(df.set_index("window")[["supervised_precision","supervised_recall","anomaly_precision","anomaly_recall"]])
    st.caption("Performance varies materially across consecutive time windows, supporting the need for drift monitoring and threshold review.")

elif page=="Explainability":
    st.header("Task 1.6 — Explainability")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Impurity-based importance")
        n=pd.DataFrame(NATIVE,columns=["Feature","Importance"])
        st.dataframe(n,hide_index=True,use_container_width=True)
        st.bar_chart(n.set_index("Feature"))
    with c2:
        st.subheader("Permutation importance")
        p=pd.DataFrame(PERM,columns=["Feature","Importance"])
        st.dataframe(p,hide_index=True,use_container_width=True)
        st.bar_chart(p.set_index("Feature"))
    st.subheader("SHAP case summary")
    st.dataframe(pd.DataFrame(SHAP_CASES),hide_index=True,use_container_width=True)
    st.write("V14 and V17 recur prominently across both global importance and local SHAP explanations.")

elif page=="Leakage Audit":
    st.header("Leakage Audit and Methodological Controls")
    st.markdown("""
- **Duplicates:** exact duplicates were handled before random splitting to prevent identical records crossing partitions.
- **Scaling:** scale-sensitive transformations were fitted on training data only or inside CV pipelines.
- **Feature analysis:** informative-feature calculations used training data only.
- **Hyperparameter tuning:** stratified 5-fold CV used training data only with PR-AUC optimization.
- **SMOTE / undersampling:** resampling was performed inside training/CV processing, not before the split.
- **Threshold selection:** validation data were used for threshold selection; the test set remained untouched until final evaluation.
- **Temporal protocol:** chronological order was preserved for the drift experiment.
- **Explainability:** feature importance and SHAP summaries come from the fitted model and actual evaluated observations.
""")
    st.subheader("Computational limitation")
    st.write("Selected tuning stages, especially k-NN, used documented training-only stratified computational subsets because repeated full-dataset 5-fold CV was computationally prohibitive. Validation and test records were not used in those tuning subsets.")

st.divider()
st.caption("Academic demonstration only. Not a production banking decision system.")
