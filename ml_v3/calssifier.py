"""
fault_classifier_app.py

Interactive, visual front-end for train_classifiers.py.

Upload your own feature CSV(s), pick the target and task, tune the models,
and explore cross-validated results, confusion matrices, feature importance,
and live predictions - no files written to disk unless you click Download.

Setup (same folder as train_classifiers__v3.py, or train_classifiers.py):
    pip install streamlit plotly scikit-learn pandas numpy joblib
    streamlit run fault_classifier_app.py
"""
import io
import json
from datetime import datetime
from functools import partial

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Reuse the constants/helpers from the original script so the two never drift.
try:
    import train_classifiers__v3 as tc
except ImportError:
    try:
        import train_classifiers as tc
    except ImportError:
        st.set_page_config(page_title="Fault classifier")
        st.error("Couldn't import your training script. Put this app in the same folder as "
                 "`train_classifiers__v3.py` (or rename that file to `train_classifiers.py`).")
        st.stop()

st.set_page_config(page_title="Fault classifier explorer", layout="wide")

MODEL_NAMES = ["RandomForest", "SVM_RBF", "KNN", "LogisticRegression"]
DELAY_COLS = ["detection_delay_s", "breaker_open_delay_s"]


# --------------------------------------------------------------------------- #
# Data + model helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_csvs(files: tuple) -> pd.DataFrame:
    frames = [pd.read_csv(io.BytesIO(data)) for _, data in files]
    return pd.concat(frames, ignore_index=True)


def prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Same imputation as the training script: NaN delay = 'never tripped'."""
    X = df[feature_cols].astype(float).copy()
    for c in DELAY_COLS:
        if c in X.columns:
            X[c] = X[c].fillna(tc.DELAY_SENTINEL)
    return X.fillna(0)


def make_pipe(name: str, p: dict, n_features: int, y: pd.Series, n_splits: int) -> Pipeline:
    if name == "RandomForest":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=p["rf_trees"], random_state=42))])
    if name == "SVM_RBF":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=p["svm_c"], probability=True, random_state=42))])
    if name == "LogisticRegression":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=p["lr_c"], random_state=42))])
    # KNN: feature selection first, neighbours capped to the smallest training fold
    n_feat = max(1, min(p["knn_features"], n_features))
    min_train_fold = int(y.value_counts().min() * (n_splits - 1) // n_splits)
    k = max(1, min(p["knn_k"], min_train_fold))
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(partial(mutual_info_classif, random_state=42), k=n_feat)),
                     ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance"))])


@st.cache_data(show_spinner="Cross-validating…")
def run_cv(X: pd.DataFrame, y: pd.Series, models: tuple, params: dict, n_splits: int) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = sorted(y.unique())
    out = {}
    for name in models:
        oof = np.empty(len(y), dtype=object)
        fold_acc = []
        for tr, te in cv.split(X, y):
            pipe = make_pipe(name, params, X.shape[1], y.iloc[tr], n_splits)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            oof[te] = pred
            fold_acc.append(accuracy_score(y.iloc[te], pred))
        rep = classification_report(y, oof, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, oof, labels=labels)
        out[name] = {
            "acc_mean": float(np.mean(fold_acc)), "acc_std": float(np.std(fold_acc)),
            "precision": rep["macro avg"]["precision"], "recall": rep["macro avg"]["recall"],
            "f1": rep["macro avg"]["f1-score"], "cm": cm, "labels": labels,
            "per_class": pd.DataFrame(rep).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore"),
        }
    return out


@st.cache_resource(show_spinner="Training final model on all rows…")
def fit_final(_X, _y, data_key: str, name: str, params_key: str, _params: dict, n_splits: int):
    pipe = make_pipe(name, _params, _X.shape[1], _y, n_splits)
    pipe.fit(_X, _y)
    return pipe


@st.cache_data(show_spinner=False)
def rf_importance(X: pd.DataFrame, y: pd.Series, trees: int) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=trees, random_state=42).fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Sidebar: data + settings
# --------------------------------------------------------------------------- #
st.title("Fault classifier explorer")

with st.sidebar:
    st.header("1 · Data")
    uploads = st.file_uploader("Feature CSV file(s)", type="csv", accept_multiple_files=True)
    if not uploads:
        st.info("Upload one or more CSVs to begin. Multiple files are concatenated.")
        st.stop()
    df = load_csvs(tuple((u.name, u.getvalue()) for u in uploads))

    st.header("2 · Task")
    all_cols = list(df.columns)
    target_col = st.selectbox("Target column", all_cols,
                              index=all_cols.index("fault_type") if "fault_type" in all_cols else 0,
                              help="Use 'mechanism' here for the coarse 6-class task.")
    raw_target = df[target_col].astype(str)
    task = st.radio("Task", ["Multiclass (target as-is)",
                             "Binary (healthy vs fault)",
                             "Merged (ground-ambiguous pairs collapsed)"])
    if task.startswith("Binary"):
        values = sorted(raw_target.unique())
        healthy = st.selectbox("Which value means healthy?", values,
                               index=values.index("Healthy") if "Healthy" in values else 0)
        y = pd.Series(np.where(raw_target == healthy, "Healthy", "Fault"), name="target")
    elif task.startswith("Merged"):
        y = tc.merged_fault_type(raw_target).rename("target")
    else:
        y = raw_target.rename("target")

    default_excl = [c for c in tc.LEAKY_COLS if c in all_cols] + [target_col]
    excluded = st.multiselect("Excluded columns (leakage / metadata)", all_cols,
                              default=sorted(set(default_excl), key=all_cols.index),
                              help="Defaults come from LEAKY_COLS in your script plus the target.")
    feature_cols = [c for c in all_cols if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    skipped_text = [c for c in all_cols if c not in excluded and c not in feature_cols]
    if skipped_text:
        st.caption(f"Non-numeric columns ignored: {', '.join(skipped_text)}")

    st.header("3 · Models")
    chosen = st.multiselect("Models to compare", MODEL_NAMES, default=MODEL_NAMES)
    params = {
        "rf_trees": st.slider("Random Forest trees", 50, 500, 200, 50),
        "svm_c": st.select_slider("SVM C", [0.1, 1, 10, 50, 100, 500], value=50),
        "lr_c": st.select_slider("Logistic Regression C", [0.01, 0.1, 1, 10, 100], value=1),
        "knn_k": st.slider("KNN neighbours", 1, 15, 3),
        "knn_features": st.slider("KNN: features kept (SelectKBest)", 1, max(2, len(feature_cols)),
                                  min(tc.KNN_N_FEATURES, max(1, len(feature_cols)))),
    }

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
counts = y.value_counts()
if not feature_cols:
    st.error("No numeric feature columns left after exclusions."); st.stop()
if not chosen:
    st.warning("Pick at least one model in the sidebar."); st.stop()
if len(counts) < 2:
    st.error(f"Only one class present ('{counts.index[0]}') - need at least two."); st.stop()
if counts.min() < 2:
    st.error(f"Class '{counts.idxmin()}' has only {counts.min()} sample(s) - need at least 2 per class."); st.stop()

X = prep_X(df, feature_cols)
max_splits = tc.safe_n_splits(y, 10)
with st.sidebar:
    n_splits = st.slider("CV folds", 2, max(2, max_splits), min(5, max(2, max_splits)),
                         help=f"Capped at the smallest class size ({counts.min()}).")

results = run_cv(X, y, tuple(chosen), params, n_splits)
best = max(results, key=lambda n: results[n]["acc_mean"])
data_key = f"{len(df)}-{int(pd.util.hash_pandas_object(X).sum())}-{int(pd.util.hash_pandas_object(y).sum())}"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Features", len(feature_cols))
c3.metric("Classes", len(counts))
c4.metric(f"Best ({n_splits}-fold CV)", best, f"{results[best]['acc_mean']:.1%}")

tab_data, tab_compare, tab_cm, tab_feat, tab_pred = st.tabs(
    ["Data", "Model comparison", "Confusion matrix", "Features", "Predict"])

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with tab_data:
    left, right = st.columns(2)
    with left:
        cdf = counts.rename_axis("class").reset_index(name="count")
        fig = px.bar(cdf, x="class", y="count", color="class", title="Class distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if len(feature_cols) >= 2:
            Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
            pdf = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "class": y.values})
            st.plotly_chart(px.scatter(pdf, x="PC1", y="PC2", color="class",
                                       title="PCA of scaled features (2D view of separability)"),
                            use_container_width=True)
    st.dataframe(df, use_container_width=True, height=250)

# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
with tab_compare:
    summary = pd.DataFrame({
        "Model": list(results),
        "Accuracy": [r["acc_mean"] for r in results.values()],
        "± std": [r["acc_std"] for r in results.values()],
        "Macro precision": [r["precision"] for r in results.values()],
        "Macro recall": [r["recall"] for r in results.values()],
        "Macro F1": [r["f1"] for r in results.values()],
    })
    fig = go.Figure(go.Bar(x=summary["Model"], y=summary["Accuracy"],
                           error_y=dict(type="data", array=summary["± std"]),
                           marker_color=["#2a9d8f" if m == best else "#3b6ea5" for m in summary["Model"]]))
    fig.update_layout(title="Mean CV accuracy (error bars = std across folds)",
                      yaxis=dict(range=[max(0, summary["Accuracy"].min() - 0.15), 1.02], tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary.style.format({c: "{:.3f}" for c in summary.columns[1:]}), use_container_width=True)
    st.download_button("Download metrics CSV", summary.to_csv(index=False), "cv_metrics.csv", "text/csv")

    model_for_pc = st.selectbox("Per-class precision / recall / F1 for", list(results), index=list(results).index(best))
    pc = results[model_for_pc]["per_class"][["precision", "recall", "f1-score", "support"]]
    st.plotly_chart(px.bar(pc.reset_index().melt(id_vars="index", value_vars=["precision", "recall", "f1-score"]),
                           x="index", y="value", color="variable", barmode="group",
                           labels={"index": "class", "value": "score", "variable": ""}),
                    use_container_width=True)

# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
with tab_cm:
    a, b = st.columns([2, 1])
    cm_model = a.selectbox("Model", list(results), index=list(results).index(best), key="cm_model")
    normalize = b.checkbox("Normalise rows (recall per class)")
    r = results[cm_model]
    cm, labels = r["cm"], r["labels"]
    show = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig = px.imshow(show, x=labels, y=labels, color_continuous_scale="Blues", text_auto=".2f" if normalize else True,
                    labels=dict(x="Predicted", y="Actual", color="Share" if normalize else "Count"))
    fig.update_layout(height=max(400, 28 * len(labels) + 150))
    st.plotly_chart(fig, use_container_width=True)
    mis = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels))
           if i != j and cm[i, j] > 0]
    if mis:
        st.subheader("Misclassifications")
        st.dataframe(pd.DataFrame(sorted(mis, key=lambda t: -t[2]), columns=["Actual", "Predicted", "Count"]),
                     use_container_width=True, hide_index=True)
    else:
        st.success("No misclassifications - perfect confusion matrix.")

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
with tab_feat:
    imp = rf_importance(X, y, params["rf_trees"])
    top_n = st.slider("Show top N features", 3, len(imp), min(15, len(imp)))
    top = imp.head(top_n)[::-1]
    st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h",
                           labels={"x": "Importance (Random Forest)", "y": ""},
                           height=max(300, 26 * len(top))), use_container_width=True)
    feat = st.selectbox("Distribution of a feature by class", list(imp.index))
    st.plotly_chart(px.box(pd.DataFrame({feat: X[feat], "class": y}), x="class", y=feat, color="class",
                           points="all"), use_container_width=True)

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
with tab_pred:
    pm = st.selectbox("Model used for prediction", list(results), index=list(results).index(best), key="pred_model")
    model = fit_final(X, y, data_key, pm, json.dumps(params, sort_keys=True), params, n_splits)
    st.caption("Fit once on 100% of your rows. Trust the CV numbers for accuracy, not training accuracy.")

    buf = io.BytesIO()
    joblib.dump(model, buf)
    schema = {"model_name": pm, "classes": sorted(y.unique().tolist()), "feature_cols_in_order": feature_cols,
              "delay_sentinel_s": tc.DELAY_SENTINEL, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    d1, d2 = st.columns(2)
    d1.download_button("Download model (.joblib)", buf.getvalue(), f"final_model_{pm}.joblib")
    d2.download_button("Download feature schema (.json)", json.dumps(schema, indent=2), f"final_model_{pm}_schema.json")

    mode = st.radio("Input", ["Edit a single sample", "Upload a CSV of new samples"], horizontal=True)

    if mode.startswith("Edit"):
        base = st.selectbox("Start from", ["Overall median"] + [f"Median of {c}" for c in sorted(y.unique())])
        row = X.median() if base == "Overall median" else X[y == base.replace("Median of ", "", 1)].median()
        edit_df = pd.DataFrame({"feature": feature_cols, "value": row[feature_cols].values})
        edited = st.data_editor(edit_df, disabled=["feature"], hide_index=True, use_container_width=True,
                                height=320, key=f"editor_{base}_{pm}")
        sample = pd.DataFrame([edited["value"].astype(float).values], columns=feature_cols)
        pred = model.predict(sample)[0]
        st.metric("Predicted class", str(pred))
        if hasattr(model, "predict_proba"):
            probs = pd.Series(model.predict_proba(sample)[0], index=model.classes_).sort_values()
            st.plotly_chart(px.bar(x=probs.values, y=probs.index, orientation="h",
                                   labels={"x": "Probability", "y": ""}, range_x=[0, 1],
                                   height=max(250, 26 * len(probs))), use_container_width=True)
    else:
        new_file = st.file_uploader("CSV with the same feature columns", type="csv", key="new_csv")
        if new_file:
            new_df = pd.read_csv(new_file)
            missing = [c for c in feature_cols if c not in new_df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                Xn = prep_X(new_df, feature_cols)
                out = new_df.copy()
                out["predicted"] = model.predict(Xn)
                if hasattr(model, "predict_proba"):
                    out["confidence"] = model.predict_proba(Xn).max(axis=1)
                st.dataframe(out, use_container_width=True)
                st.plotly_chart(px.histogram(out, x="predicted", color="predicted", title="Predicted class counts"),
                                use_container_width=True)
                if target_col in new_df.columns and task.startswith("Multiclass"):
                    acc = accuracy_score(new_df[target_col].astype(str), out["predicted"])
                    st.metric("Accuracy against the target column in this file", f"{acc:.1%}")
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")