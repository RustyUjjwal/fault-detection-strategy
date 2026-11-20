"""
train_classifiers_v3.py

Trains and compares classical ML classifiers on the feature CSV(s) exported
from the Simulink model, for four tasks:
  - Binary:      Healthy vs Fault (detector/trip outputs are EXCLUDED here,
                 because Healthy never trips and every fault does, so those
                 columns would just hand the model the label)
  - Mechanism:   6 classes (which physical phenomenon)
  - Multiclass:  28 fault types (mechanism + which phases/ground)
  - Merged:      27 classes, LLLG folded into LLL (the only pair that is
                 physically identical: balanced sources keep the star point
                 at 0 V, so grounding changes nothing)

Models compared: Random Forest, SVM (RBF), K-Nearest Neighbours, Logistic
Regression - each in a StandardScaler pipeline, evaluated with stratified
k-fold cross-validation (not a single train/test split - the dataset this
was built for is small, and a single split is too noisy to trust).

Outputs (written to --output-dir, default ml_output/):
  - ml_report.md              human-readable write-up of everything below
  - confusion_binary_<model>.png
  - confusion_multiclass_<model>.png
  - feature_importance.png    from the Random Forest multiclass model

Usage:
    python train_classifiers_v3.py fault_detection_features_v3.csv
    python train_classifiers_v3.py a.csv b.csv -o ml_output
"""
import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# Columns that are experiment design metadata, not physical measurements -
# excluded from features because they'd leak the label (e.g. severity is a
# fixed value per fault type by construction, not something a real deployed
# system would get to see; noise_seed (v6+) is assigned in a fixed,
# label-correlated order by the sweep script, so it would leak the class
# through seed-number ordering rather than any real signal property).
LEAKY_COLS = ['fault_type', 'label', 'severity', 'fault_onset_s', 'fault_duration_s',
              'noise_seed', 'mechanism', 'G', 'A', 'B', 'C']

# NaN in these columns means "never happened" (no trip), which is itself
# meaningful - it's imputed with a large sentinel rather than dropped/mean-
# filled, so the classifier can use "never tripped" as a real signal.
DETECTOR_KEYS = ['cusum', 'residual', 'detection_delay', 'breaker_open', 'tripped', 'letthrough']
DELAY_COLS = ['detection_delay_s_agg']   # v8 has no breaker_open_delay_s
DELAY_SENTINEL = 5.0   # seconds - comfortably beyond any real detection/open delay in this project

# KNN-only: number of features SelectKBest keeps before distance is computed.
KNN_N_FEATURES = 20

MODELS = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=50, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance'),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
}


def load_data(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c not in LEAKY_COLS]
    X = df[feature_cols].copy()
    # v8: max_cusum_* / max_Cusum_* sit at their cap in every faulty run, drop zero-variance columns
    const = [c for c in feature_cols if X[c].nunique(dropna=True) <= 1]
    if const:
        print(f'Dropping {len(const)} constant columns: {const}')
        feature_cols = [c for c in feature_cols if c not in const]
        X = X[feature_cols]
    # letthrough_* are NaN when the contacts never open (Healthy): keep that as a large-free zero

    for col in DELAY_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(DELAY_SENTINEL)
    X = X.fillna(0)   # any other stray NaN - shouldn't occur, but don't crash on it
    return X, feature_cols


def safe_n_splits(y: pd.Series, max_splits: int = 5) -> int:
    min_class_count = y.value_counts().min()
    return max(2, min(max_splits, min_class_count))


def evaluate_task(X: pd.DataFrame, y: pd.Series, task_name: str, out_dir: str) -> dict:
    results = {}
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        results['skipped'] = (
            f"Skipped {task_name}: only one class present ('{class_counts.index[0]}') - "
            "need at least two classes to train a classifier."
        )
        return results
    if class_counts.min() < 2:
        results['skipped'] = (
            f"Skipped {task_name}: class '{class_counts.idxmin()}' has only "
            f"{class_counts.min()} sample(s) - need at least 2 per class for cross-validation."
        )
        return results

    n_splits = safe_n_splits(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels_sorted = sorted(y.unique())

    per_model = {}
    for name, clf in MODELS.items():
        if name == 'KNN':
            # Cap SelectKBest's k at the number of available features so this
            # doesn't break on a future CSV with fewer columns than KNN_N_FEATURES.
            n_feat = max(1, min(KNN_N_FEATURES, X.shape[1]))
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('select', SelectKBest(mutual_info_classif, k=n_feat)),
                ('clf', clf),
            ])
            # n_neighbors still needs to stay <= smallest training fold size
            min_train_fold = class_counts.min() * (n_splits - 1) // n_splits
            k = max(1, min(3, int(min_train_fold)))
            pipe.set_params(clf__n_neighbors=k)
        else:
            pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)])

        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=labels_sorted)

        # Off-diagonal confusion pairs, sorted by count - so the report text can
        # say e.g. "Overcurrent predicted as Overvoltage: 3" without opening a PNG.
        confusions = []
        for i, true_label in enumerate(labels_sorted):
            for j, pred_label in enumerate(labels_sorted):
                if i != j and cm[i, j] > 0:
                    confusions.append((true_label, pred_label, int(cm[i, j])))
        confusions.sort(key=lambda x: -x[2])

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=45, ha='right')
        ax.set_yticklabels(labels_sorted)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{task_name} — {name}')
        for i in range(len(labels_sorted)):
            for j in range(len(labels_sorted)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fname = f'confusion_{task_name.lower()}_{name}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

        per_model[name] = {
            'acc_mean': scores.mean(),
            'acc_std': scores.std(),
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'confusion_png': fname,
            'n_splits': n_splits,
            'confusion_matrix': cm.tolist(),
            'labels_sorted': labels_sorted,
            'confusions': confusions,
        }

    results['models'] = per_model
    results['class_counts'] = class_counts.to_dict()
    results['n_splits'] = n_splits
    return results


def best_model_name(results: dict) -> str | None:
    """Pick the model with the highest mean CV accuracy from evaluate_task()'s
    output - same "best by mean CV accuracy" logic write_report() already
    uses for the table, factored out so save_final_model() can reuse it
    instead of a hardcoded model name."""
    if 'models' not in results or not results['models']:
        return None
    return max(results['models'].items(), key=lambda kv: kv[1]['acc_mean'])[0]


def feature_importance_plot(X: pd.DataFrame, y: pd.Series, out_dir: str) -> tuple[str, pd.Series]:
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(importances))))
    importances.plot.barh(ax=ax, color='#3b6ea5')
    ax.set_xlabel('Importance (Random Forest, multiclass)')
    fig.tight_layout()
    fname = 'feature_importance.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)
    return fname, importances.sort_values(ascending=False)


def build_pipeline_for(name: str, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Build the same pipeline shape evaluate_task() uses for a given model
    name, so the saved model is trained the identical way it was cross-
    validated (same scaling, same KNN feature-selection/neighbor-count
    logic) - not just a bare sklearn estimator fit on raw columns."""
    clf = MODELS[name]
    if name == 'KNN':
        n_feat = max(1, min(KNN_N_FEATURES, X.shape[1]))
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k=n_feat)),
            ('clf', clf),
        ])
        class_counts = y.value_counts()
        n_splits = safe_n_splits(y)
        min_train_fold = class_counts.min() * (n_splits - 1) // n_splits
        k = max(1, min(3, int(min_train_fold)))
        pipe.set_params(clf__n_neighbors=k)
    else:
        pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)])
    return pipe


def save_final_model(X: pd.DataFrame, y: pd.Series, feature_cols: list[str],
                      model_name: str, task_name: str, out_dir: str) -> dict:
    """Fit `model_name` on ALL of X/y (no cross-validation split held back -
    CV already happened in evaluate_task() and produced the accuracy numbers
    in the report; this is a separate, final fit whose only job is to
    produce a deployable artifact) and save it to disk with joblib, plus a
    JSON schema describing exactly what the model expects as input.

    Returns a dict of the files written, so write_report() can mention them.
    """
    pipe = build_pipeline_for(model_name, X, y)
    pipe.fit(X, y)   # full dataset - this is NOT the CV-scored model, it's the deployable one

    model_fname = f'final_model_{task_name.lower()}_{model_name}.joblib'
    model_path = os.path.join(out_dir, model_fname)
    joblib.dump(pipe, model_path)

    schema = {
        'model_name': model_name,
        'task': task_name,
        'feature_cols_in_order': feature_cols,
        'classes': sorted(y.unique().tolist()),
        'n_training_rows': int(len(y)),
        'delay_sentinel_s': DELAY_SENTINEL,
        'delay_sentinel_cols': DELAY_COLS,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': ('This model was fit on 100% of the training rows (no held-out '
                 'split at this step). Trust the CV accuracy in ml_report.md for '
                 'a generalization estimate - do not re-score this exact model on '
                 'its own training data and report that number as accuracy.'),
    }
    schema_fname = f'final_model_{task_name.lower()}_{model_name}_schema.json'
    schema_path = os.path.join(out_dir, schema_fname)
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)

    return {'model_file': model_fname, 'schema_file': schema_fname, 'model_name': model_name}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(lines)


def write_report(paths, df, feature_cols, binary_results, multi_results, importance_png, importances,
                  out_dir, saved_models=None, mechanism_results=None, merged_results=None):
    lines = []
    lines.append('# Fault Detection — ML Classifier Comparison')
    lines.append('')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append('## Data')
    lines.append('')
    lines.append('Source file(s):')
    for p in paths:
        lines.append(f'- `{p}`')
    lines.append('')
    lines.append(f'Total rows: **{len(df)}**')
    lines.append('')
    if 'fault_type' in df.columns:
        counts = df['fault_type'].value_counts()
        lines.append(md_table(['Class', 'Count'], [[k, v] for k, v in counts.items()]))
    lines.append('')
    lines.append('Features used (design/metadata columns — `severity`, `fault_onset_s`, '
                  '`fault_duration_s`, `fault_type`/`label` — excluded as label leakage):')
    lines.append('')
    lines.append(', '.join(f'`{c}`' for c in feature_cols))
    lines.append('')

    lines.append('## ⚠ Caveats — read before trusting these numbers')
    lines.append('')
    small_dataset_note = ('Accuracy on a dataset this size can look excellent (or terrible) by '
                           'chance. Treat these numbers as a pipeline demonstration, not a fully '
                           'validated result, until the dataset is meaningfully larger.'
                           if len(df) < 500 else
                           'This is a moderate-sized dataset for the number of classes - still '
                           'worth a wider held-out test if the numbers below are going in a report.')
    lines.append(f'- **Dataset size.** {len(df)} rows across {df["fault_type"].nunique()} classes. '
                  f'{small_dataset_note}')

    # Measurement noise - detect from the DATA, not from noise_seed. noise_seed is only a
    # run ID unless the model actually has a noise source wired in; counting distinct seeds
    # (the old check) reported "independent noise" even when the seed had no effect.
    # Healthy rows have a fixed, known truth, so their post-fault Irms should be constant
    # unless real noise is present.
    noise_present = None
    healthy = df[df['fault_type'] == 'Healthy'] if 'fault_type' in df.columns else df.iloc[0:0]
    if len(healthy) > 1 and 'mean_Irms_postfault_A' in healthy.columns:
        col = healthy['mean_Irms_postfault_A']
        noise_present = bool((col.std() / max(abs(col.mean()), 1e-12)) > 1e-6)
    if noise_present is False:
        lines.append('- **No measurement noise in this dataset.** Healthy runs differ in post-fault '
                      'Irms by only ~1e-11 (relative), so `noise_seed` is just a run ID, not a real '
                      'noise realization. Rows within a class differ through severity and '
                      'onset-time windows only. Treat the near-perfect accuracies below as '
                      'results on clean, deterministic simulated data, not as robustness to noisy '
                      'sensors.')
        if len(healthy) > 1:
            feat_like = [c for c in healthy.columns if c.endswith(('_postfault_A', '_postfault_B', '_postfault_C', '_agg'))]
            n_unique = healthy[feat_like].round(4).drop_duplicates().shape[0]
            lines.append(f'- **Healthy rows are near-duplicates.** {len(healthy)} Healthy rows give '
                          f'only {n_unique} distinct post-fault feature vectors (rounded to 4 dp), '
                          'because onset time barely changes a noise-free healthy signal. Healthy '
                          'recall is therefore easier than the row count suggests.')
    elif noise_present:
        lines.append('- **Measurement noise present.** Healthy runs show run-to-run variation in '
                      'post-fault Irms, so samples within a class are independent noisy draws.')
    else:
        lines.append('- **Noise not assessed.** Could not check Healthy-row variation in this CSV.')

    # Severity range - same detect-don't-assume treatment.
    if 'severity' in df.columns:
        spread = df.groupby('fault_type')['severity'].nunique()
        varying_classes = spread[spread > 1].index.tolist()
        if varying_classes:
            lines.append('- **Severity varies within a range** for '
                          f'{", ".join(varying_classes)} (drawn per-run, not fixed), which is a '
                          'more realistic test of separability than one severity value per class.')

    lines.append('- **Binary task excludes detector/trip outputs** (CUSUM, residual, '
                  'detection delay, breaker, tripped). Healthy never trips and every fault does, '
                  'so including them would let the model read the label from the detector.')
    lines.append('- **Engineered, not raw, features.** Inputs like `max_cusum` and `max_residual` '
                  'are already outputs of the CUSUM/Luenberger detectors, not raw signal '
                  'measurements — so this model is partly learning to combine existing detector '
                  'outputs rather than replacing them outright.')
    lines.append('- `detection_delay_s_agg` / `breaker_open_delay_s` are NaN for runs that never '
                  f'tripped (Healthy, or a missed detection) and were imputed with a {DELAY_SENTINEL}s '
                  'sentinel to represent "never" — a modelling choice, not a measured value.')
    lines.append('')

    def task_section(title, results):
        lines.append(f'## {title}')
        lines.append('')
        if 'skipped' in results:
            lines.append(f'*{results["skipped"]}*')
            lines.append('')
            return
        lines.append(f'{results["n_splits"]}-fold stratified cross-validation.')
        lines.append('')
        header = ['Model', 'Accuracy (mean ± std)', 'Macro precision', 'Macro recall', 'Macro F1']
        rows = []
        best_model, best_acc = None, -1
        for name, m in results['models'].items():
            rows.append([name, f'{m["acc_mean"]:.3f} ± {m["acc_std"]:.3f}',
                         f'{m["macro_precision"]:.3f}', f'{m["macro_recall"]:.3f}',
                         f'{m["macro_f1"]:.3f}'])
            if m['acc_mean'] > best_acc:
                best_acc, best_model = m['acc_mean'], name
        lines.append(md_table(header, rows))
        lines.append('')
        lines.append(f'Best by mean CV accuracy: **{best_model}** ({best_acc:.3f}).')
        lines.append('')
        lines.append('Confusion matrices:')
        lines.append('')
        for name, m in results['models'].items():
            lines.append(f'### {name}')
            lines.append('')
            lines.append(f'`{m["confusion_png"]}`')
            lines.append('')
            # Raw counts as a markdown table (rows = actual, cols = predicted) so the
            # numbers are readable directly from the .md, not just the PNG.
            cm_header = ['Actual \\ Predicted'] + m['labels_sorted']
            cm_rows = []
            for i, row_label in enumerate(m['labels_sorted']):
                cm_rows.append([row_label] + [m['confusion_matrix'][i][j] for j in range(len(m['labels_sorted']))])
            lines.append(md_table(cm_header, cm_rows))
            lines.append('')
            if m['confusions']:
                lines.append('Misclassifications (actual → predicted, count):')
                lines.append('')
                for true_label, pred_label, count in m['confusions']:
                    lines.append(f'- {true_label} → {pred_label}: {count}')
            else:
                lines.append('No misclassifications — perfect confusion matrix.')
            lines.append('')

    task_section('Binary — Healthy vs Fault', binary_results)
    if mechanism_results:
        task_section('Mechanism — Coarse (6-class: which physical phenomenon)', mechanism_results)
    task_section('Multiclass — Fault Type (fine-grained: which phase(s)/mechanism)', multi_results)
    if merged_results:
        lines.append('')
        lines.append('**Why a merged target**: with balanced three-phase sources the '
                      'star point of the fault network sits at 0 V, so grounding an LLL fault '
                      'changes nothing and LLL / LLLG are the same simulation under two '
                      'labels (identical I0, near-identical Irms). LL vs LLG are kept apart: '
                      'the ground return carries zero-sequence current, so `mean_I0_rms_postfault` '
                      'separates them.')
        task_section('Multiclass — Merged (LLLG folded into LLL, 27-class)', merged_results)

    if saved_models:
        lines.append('## Saved Models (deployable artifacts)')
        lines.append('')
        lines.append('Unlike the cross-validated models above (each fold trains-then-discards a '
                      'model - that\'s what makes CV a fair accuracy estimate), these were fit '
                      '**once, on 100% of the rows**, and saved to disk so they can be reloaded '
                      'and used on new data without retraining. Treat the CV numbers above as the '
                      'accuracy estimate for these models - not their (meaningless) training accuracy.')
        lines.append('')
        for task, info in saved_models.items():
            lines.append(f'- **{task}**: `{info["model_name"]}` saved to `{info["model_file"]}` '
                          f'(feature schema: `{info["schema_file"]}`)')
        lines.append('')
        lines.append('To reuse:')
        lines.append('```python')
        lines.append('import joblib, json')
        lines.append("model = joblib.load('final_model_multiclass_<ModelName>.joblib')")
        lines.append("schema = json.load(open('final_model_multiclass_<ModelName>_schema.json'))")
        lines.append("X_new = new_df[schema['feature_cols_in_order']].fillna(0)  # same prep as training")
        lines.append('preds = model.predict(X_new)')
        lines.append('```')
        lines.append('')

    if importance_png:
        lines.append('## Feature Importance (Random Forest, multiclass)')
        lines.append('')
        lines.append(f'![feature importance]({importance_png})')
        lines.append('')
        lines.append('Ranked (highest first):')
        lines.append('')
        lines.append(md_table(['Feature', 'Importance'],
                               [[feat, f'{val:.4f}'] for feat, val in importances.items()]))
        lines.append('')

    with open(os.path.join(out_dir, 'ml_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# Only LLLG/LLL are physically identical. LLG vs LL differ in zero-sequence
# current (ground return), so they stay separate classes.
GROUND_AMBIGUOUS_MERGE = {'LLLG': 'LLL'}


def merged_fault_type(series):
    return series.replace(GROUND_AMBIGUOUS_MERGE)


def main():
    parser = argparse.ArgumentParser(description='Compare classical ML classifiers on fault_detection_features CSV(s).')
    parser.add_argument('csv_files', nargs='+', help='One or more feature CSV files')
    parser.add_argument('-o', '--output-dir', default='ml_output', help='Directory for report + plots (default: ml_output)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.csv_files)
    X, feature_cols = build_features(df)

    print(f'Loaded {len(df)} rows from {len(args.csv_files)} file(s).')
    print(f'Classes: {sorted(df["fault_type"].unique())}')

    binary_results = {}
    if 'fault_type' in df.columns:
        y_binary = (df['fault_type'] != 'Healthy').map({True: 'Fault', False: 'Healthy'})
        binary_cols = [c for c in feature_cols if not any(k in c for k in DETECTOR_KEYS)]
        X_binary = X[binary_cols]
        binary_results = evaluate_task(X_binary, y_binary, 'Binary', args.output_dir)

    mechanism_results = {}
    y_mechanism = None
    if 'mechanism' in df.columns:
        y_mechanism = df['mechanism']
        mechanism_results = evaluate_task(X, y_mechanism, 'Mechanism', args.output_dir)

    y_multi = df['fault_type']
    multi_results = evaluate_task(X, y_multi, 'Multiclass', args.output_dir)

    y_merged = None
    merged_results = {}
    if 'fault_type' in df.columns:
        y_merged = merged_fault_type(df['fault_type'])
        merged_results = evaluate_task(X, y_merged, 'MulticlassMerged', args.output_dir)

    importance_png = None
    importances = None
    if multi_results.get('models'):
        importance_png, importances = feature_importance_plot(X, y_multi, args.output_dir)

    saved_models = {}
    if binary_results.get('models'):
        name = best_model_name(binary_results)
        saved_models['Binary'] = save_final_model(X_binary, y_binary, binary_cols, name, 'Binary', args.output_dir)
        print(f'Saved binary model: {name} -> {saved_models["Binary"]["model_file"]}')
    if mechanism_results.get('models'):
        name = best_model_name(mechanism_results)
        saved_models['Mechanism'] = save_final_model(X, y_mechanism, feature_cols, name, 'Mechanism', args.output_dir)
        print(f'Saved mechanism model: {name} -> {saved_models["Mechanism"]["model_file"]}')
    if multi_results.get('models'):
        name = best_model_name(multi_results)
        saved_models['Multiclass'] = save_final_model(X, y_multi, feature_cols, name, 'Multiclass', args.output_dir)
        print(f'Saved multiclass model: {name} -> {saved_models["Multiclass"]["model_file"]}')
    if merged_results.get('models'):
        name = best_model_name(merged_results)
        saved_models['MulticlassMerged'] = save_final_model(X, y_merged, feature_cols, name, 'MulticlassMerged', args.output_dir)
        print(f'Saved merged-multiclass model: {name} -> {saved_models["MulticlassMerged"]["model_file"]}')

    write_report(args.csv_files, df, feature_cols, binary_results, multi_results,
                 importance_png, importances, args.output_dir, saved_models,
                 mechanism_results=mechanism_results, merged_results=merged_results)

    print(f'Wrote report + plots to {args.output_dir}/')


if __name__ == '__main__':
    main()