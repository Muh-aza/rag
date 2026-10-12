"""Sequence-based ML baselines for kinase-substrate phosphorylation prediction.
5-fold StratifiedGroupKFold grouped by substrate UniProt ID (no substrate shared between train and test)."""
import sys, json, itertools, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
from lightgbm import LGBMClassifier

SEED = 42
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "combined_with_label.csv")).reset_index(drop=True)
OUTDIR = os.path.join(ROOT, "results")
AA = "ACDEFGHIKLMNPQRSTVWY"; DP = ["".join(p) for p in itertools.product(AA, AA)]
def comp(seq):
    seq = "".join(c for c in seq.upper() if c in AA); n = max(len(seq), 1)
    aac = np.array([seq.count(a) for a in AA]) / n
    idx = {p: i for i, p in enumerate(DP)}; dpc = np.zeros(400)
    for i in range(len(seq) - 1): dpc[idx[seq[i:i+2]]] += 1
    dpc /= max(len(seq) - 1, 1)
    return np.concatenate([aac, dpc])
cache = {}
def feat(s):
    if s not in cache: cache[s] = comp(s)
    return cache[s]
X = np.hstack([np.vstack(d.Sequence.map(feat)), np.vstack(d["Kinase Domain"].map(feat))])
y = d.label.values; groups = d.Uniprotid.values
print("X", X.shape, "pos", y.sum())

models = {
  "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)),
  "Random Forest": lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED),
  "LightGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=SEED, verbose=-1, n_jobs=2),
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y, groups))
rows, oof_all = [], {}
def metrics(yt, yp, pr):
    return dict(precision=precision_score(yt, yp, zero_division=0), recall=recall_score(yt, yp), f1=f1_score(yt, yp),
                accuracy=accuracy_score(yt, yp), mcc=matthews_corrcoef(yt, yp), auroc=roc_auc_score(yt, pr))
# kinase-prior baseline: per-kinase positive rate in training folds (sanity check for label bias)
oof = np.zeros(len(y))
for tr, te in folds:
    rate = pd.Series(y[tr]).groupby(d.kinase.values[tr]).mean(); glob = y[tr].mean()
    oof[te] = d.kinase.iloc[te].map(rate).fillna(glob).values
oof_all["Kinase prior"] = oof
for name, mk in models.items():
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = mk(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    oof_all[name] = oof
    print("done", name, flush=True)
for name, oof in oof_all.items():
    pooled = metrics(y, (oof >= 0.5).astype(int), oof)
    per = [metrics(y[te], (oof[te] >= 0.5).astype(int), oof[te]) for _, te in folds]
    r = {"model": name}
    for k in pooled: r[k] = pooled[k]; r[k + "_sd"] = float(np.std([p[k] for p in per], ddof=1))
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUTDIR, "baseline_results.csv"), index=False)
pd.DataFrame(oof_all).assign(Uniprotid=d.Uniprotid, kinase=d.kinase, label=y).to_csv(os.path.join(OUTDIR, "baseline_oof_predictions.csv"), index=False)
json.dump({"n_pairs": int(len(y)), "n_pos": int(y.sum()), "n_sub": int(d.Uniprotid.nunique()), "n_kin": int(d.kinase.nunique()),
           "fold_sizes": [int(len(te)) for _, te in folds], "fold_pos_rate": [float(y[te].mean()) for _, te in folds]}, open(os.path.join(OUTDIR, "cv_info.json"), "w"), indent=1)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
