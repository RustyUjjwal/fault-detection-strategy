"""
robustness_checks.py

Extra sanity checks on fault_detection_features.csv, beyond the plain
stratified CV in train_classifiers.py. Reproduces the numbers quoted in the
README's "Additional checks" section.

  1. Severity-band holdout: within each class, train on the lowest 60% of
     severity and test on the top 40% (Healthy: split by onset time instead,
     since its severity is a dummy value).
  2. Detector ablation: drop every CUSUM / residual / detection-delay / trip
     column and re-run CV on the 24-class merged task.
  3. Noise check: spread of Healthy post-fault Irms (zero spread means no
     measurement noise in the simulation).
  4. Ceiling for the LL vs LLG pair: best possible accuracy from fault
     resistance alone, given severity ~ U(0.3, 0.8) and R_eff = 2s/(s+2).

Usage:
    python robustness_checks.py [fault_detection_features.csv]
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

LEAKY_COLS = ['fault_type', 'label', 'severity', 'fault_onset_s', 'fault_duration_s',
              'noise_seed', 'mechanism', 'G', 'A', 'B', 'C']
MERGE = {'LLG_AB': 'LL_AB', 'LLG_BC': 'LL_BC', 'LLG_CA': 'LL_CA', 'LLLG': 'LLL'}


def main(path):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c not in LEAKY_COLS]
    df['merged'] = df['fault_type'].replace(MERGE)
    print(f'{len(df)} rows, {len(feat)} features\n')

    # 1. Severity-band holdout
    df['sev_rank'] = df.groupby('fault_type')['severity'].rank(pct=True)
    healthy = df['fault_type'] == 'Healthy'
    train = df[((df['sev_rank'] <= 0.6) & ~healthy) | (healthy & (df['fault_onset_s'] <= 0.8))]
    test = df[((df['sev_rank'] > 0.6) & ~healthy) | (healthy & (df['fault_onset_s'] > 0.8))]
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=0),
        'LogisticRegression': Pipeline([('s', StandardScaler()),
                                        ('m', LogisticRegression(max_iter=3000))]),
    }
    print('1. Severity-band holdout (train low 60% severity, test top 40%)')
    for task in ('mechanism', 'merged'):
        for name, model in models.items():
            model.fit(train[feat], train[task])
            acc = (model.predict(test[feat]) == test[task]).mean()
            print(f'   {task:9s} {name:18s} accuracy = {acc:.3f} (n_test={len(test)})')

    # 2. Detector ablation
    detector = [c for c in feat if any(k in c for k in
                ['cusum', 'residual', 'detection_delay', 'breaker_open', 'tripped'])]
    physics = [c for c in feat if c not in detector]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print('\n2. Detector ablation, RandomForest, merged 24-class, 5-fold CV')
    for name, cols in (('all features', feat), ('without detector/trip outputs', physics)):
        score = cross_val_score(RandomForestClassifier(200, random_state=0),
                                df[cols], df['merged'], cv=cv).mean()
        print(f'   {name:30s} ({len(cols)} cols): {score:.3f}')

    # 3. Noise check
    h = df.loc[healthy, 'mean_Irms_postfault_A']
    rel = h.std() / abs(h.mean())
    print(f'\n3. Healthy post-fault Irms (phase A): relative std = {rel:.2e} '
          f'-> {"no measurement noise" if rel < 1e-6 else "noise present"}')

    # 4. LL vs LLG ceiling
    rng = np.random.default_rng(0)
    s = rng.uniform(0.3, 0.8, 400000)
    r_ll, r_llg = s, 2 * s / (s + 2)
    bins = np.linspace(0.25, 0.8, 111)
    h1, _ = np.histogram(r_ll, bins, density=True)
    h2, _ = np.histogram(r_llg, bins, density=True)
    ceiling = 0.5 * np.sum(np.maximum(h1, h2)) * (bins[1] - bins[0])
    print(f'\n4. Best possible LL-vs-LLG accuracy from fault resistance alone: {ceiling:.3f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'fault_detection_features.csv')
