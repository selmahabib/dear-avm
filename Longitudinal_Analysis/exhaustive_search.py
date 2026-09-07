"""
Exhaustive feature subset search (sensitivity analysis on feature selection)
"""
import os
import warnings
from datetime import datetime
from itertools import combinations
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import cumulative_dynamic_auc
from survival_data import CANDIDATE_FEATURES, FINAL_FEATURES, load_and_engineer, surv_array
from survival_helpers import build_pipe, cindex, nested_cv_risk

# Settings
DATA_PATH = r'R:\cerebral_avm\Precision_Medicine_Query_Optimized_2_deident.xlsx'
SEED = 42
MIN_SIZE = 2
MAX_SIZE = 12
SCREEN_ALPHA = 1.0        # single alpha for the fast ranking pass
SCREEN_FOLDS = 3
TOP_N = 50                # how many subsets go through nested CV
N_OUTER = 5
N_INNER = 3
N_JOBS = -1
AUC_TIMES = np.array([1.0, 2.0, 3.0, 5.0], dtype=float)
np.random.seed(SEED)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Registry not found - check DATA_PATH at the top of this file.")
_src = os.path.dirname(os.path.abspath(DATA_PATH))
RESULTS_DIR = os.path.join(
    _src, f"bavm_exhaustive_search_{datetime.now():%Y%m%d_%H%M}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(RESULTS_DIR)
df = load_and_engineer(DATA_PATH)
y = surv_array(df)
ev = df["event"].to_numpy().astype(int)
pool = [f for f in CANDIDATE_FEATURES if f in df.columns]
print(f"Candidate pool ({len(pool)}): {pool}")
subsets = [c for k in range(MIN_SIZE, min(MAX_SIZE, len(pool)) + 1)
           for c in combinations(pool, k)]
print(f"Subsets of size {MIN_SIZE}-{MAX_SIZE}: {len(subsets)}")

def screen(feats):
    """Fast pass"""
    X = df[list(feats)].astype(float)
    skf = StratifiedKFold(n_splits=SCREEN_FOLDS, shuffle=True,
                          random_state=SEED)
    risk = np.full(len(X), np.nan)
    for tr, ts in skf.split(X, ev):
        try:
            p = build_pipe(SCREEN_ALPHA).fit(X.iloc[tr], y[tr])
            risk[ts] = p.predict(X.iloc[ts])
        except Exception:
            return {"features": feats, "mean_auc": np.nan, "c": np.nan}
    if np.isnan(risk).any() or np.std(risk) < 1e-10:
        return {"features": feats, "mean_auc": np.nan, "c": np.nan}
    try:
        t = AUC_TIMES[AUC_TIMES < df["time"].max()]
        auc, mean_auc = cumulative_dynamic_auc(y, y, risk, t)
    except Exception:
        return {"features": feats, "mean_auc": np.nan, "c": np.nan}
    return {"features": feats, "mean_auc": float(mean_auc),
            "c": cindex(df["event"], df["time"], risk)}

screened = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(screen)(f) for f in subsets)
scr = pd.DataFrame(screened).dropna(subset=["mean_auc"])
scr["k"] = scr["features"].apply(len)
scr["features"] = scr["features"].apply(lambda t: ", ".join(t))
scr = scr.sort_values("mean_auc", ascending=False).reset_index(drop=True)
scr.to_csv("exhaustive_screening_all.csv", index=False)
print(f"Ranked {len(scr)} subsets, saved exhaustive_screening_all.csv")

# Best subset at each size
plateau = scr.loc[scr.groupby("k")["mean_auc"].idxmax()].sort_values("k")
plateau.to_csv("exhaustive_plateau_by_size.csv", index=False)
print("\nBest subset at each size:")
print(plateau[["k", "mean_auc", "c", "features"]].to_string(index=False))

# Re-score the top N under nested CV
rows = []
for i, r in scr.head(TOP_N).iterrows():
    feats = [f.strip() for f in r["features"].split(",")]
    try:
        risk, fold_c, alphas = nested_cv_risk(
            df[feats].astype(float), y, ev,
            n_outer=N_OUTER, n_inner=N_INNER, seed=SEED)
        rows.append({"features": r["features"], "k": len(feats),
                     "screen_auc": round(r["mean_auc"], 4),
                     "nested_C": round(cindex(df["event"], df["time"], risk), 4),
                     "fold_C_sd": round(float(np.nanstd(fold_c, ddof=1)), 4)})
    except Exception as e:
        print(f"  {r['features']}: {type(e).__name__}")

resc = pd.DataFrame(rows).sort_values("nested_C", ascending=False)
resc.to_csv("exhaustive_top_nested.csv", index=False)
print(resc.head(15).to_string(index=False))

# The chosen model score
base_risk, base_fold_c, _ = nested_cv_risk(
    df[FINAL_FEATURES].astype(float), y, ev,
    n_outer=N_OUTER, n_inner=N_INNER, seed=SEED)
base_c = cindex(df["event"], df["time"], base_risk)
best = resc.iloc[0]
print(f"\nChosen 3-feature model: nested C = {base_c:.4f}")
print(f"Exhaustive winner:         nested C = {best['nested_C']:.4f} "
      f"({best['features']})")
print(f"Difference:                {best['nested_C'] - base_c:+.4f}")

pd.DataFrame([
    {"model": "3-feature Cox", "features": ", ".join(FINAL_FEATURES),
     "nested_C": round(base_c, 4)},
    {"model": "exhaustive winner", "features": best["features"],
     "nested_C": best["nested_C"]},
    {"model": "difference", "features": "-",
     "nested_C": round(best["nested_C"] - base_c, 4)},
]).to_csv("exhaustive_vs_published.csv", index=False)

import glob
print(f"\n{len(glob.glob('*.csv'))} files in {RESULTS_DIR}")
