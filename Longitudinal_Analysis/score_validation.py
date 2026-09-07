"""
Cross-validated integer score
"""
import os
import warnings
from datetime import datetime

from Survival import SEED
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import StratifiedKFold
from survival_data import FINAL_FEATURES, load_and_engineer, surv_array
from survival_helpers import (
    ALPHA_GRID, annual_rate, bootstrap_cindex, build_pipe, cindex,
    derive_integer_score, integer_score, score_tier,
)

# Settings
DATA_PATH = r'path/to/your/data.xlsx'
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 1000
PUBLISHED_AGE_CUTS = (27, 45, 63)
np.random.seed(SEED)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Registry not found - check DATA_PATH at the top of this file.")
_src = os.path.dirname(os.path.abspath(DATA_PATH))
RESULTS_DIR = os.path.join(
    _src, f"bavm_score_validation_{datetime.now():%Y%m%d_%H%M}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(RESULTS_DIR)
df = load_and_engineer(DATA_PATH)
y = surv_array(df)
ev = df["event"].to_numpy().astype(int)
X = df[FINAL_FEATURES].astype(float)
outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
oof_score = np.full(len(df), np.nan)
fold_rows = []
for k, (tr, ts) in enumerate(outer.split(X, y=ev), start=1):
    # Pick alpha on the training fold only
    best_a, best_s = ALPHA_GRID[0], -np.inf
    for a in ALPHA_GRID:
        cs = []
        for ti, vi in inner.split(X.iloc[tr], ev[tr]):
            try:
                p = build_pipe(a).fit(X.iloc[tr].iloc[ti], y[tr][ti])
                cs.append(cindex([s[0] for s in y[tr][vi]],
                                 [s[1] for s in y[tr][vi]],
                                 p.predict(X.iloc[tr].iloc[vi])))
            except Exception:
                continue
        cs = [c for c in cs if not np.isnan(c)]
        if cs and np.mean(cs) > best_s:
            best_s, best_a = np.mean(cs), a

    pipe = build_pipe(best_a).fit(X.iloc[tr], y[tr])
    scl = pipe.named_steps["scl"]
    age_i = FINAL_FEATURES.index("Age_dx")

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        cuts, points = derive_integer_score(
        pipe.named_steps["model"].coef_, FINAL_FEATURES,
            age_mean=float(scl.mean_[age_i]),
            age_sd=float(scl.scale_[age_i]))

    # Score the held-out fold with this fold's own brackets
    s_ts = integer_score(df.iloc[ts], age_cuts=tuple(cuts))
    oof_score[ts] = s_ts.to_numpy()

    c_fold = cindex(df.iloc[ts]["event"], df.iloc[ts]["time"], s_ts)
    fold_rows.append({
        "Fold": k, "alpha": best_a, "n_test": len(ts),
        "events_test": int(df.iloc[ts]["event"].sum()),
        "age_cut1": round(float(cuts[0]), 1),
        "age_cut2": round(float(cuts[1]), 1),
        "age_cut3": round(float(cuts[2]), 1),
        "score_C_holdout": round(c_fold, 4),
    })
    print(f"  fold {k}: alpha={best_a}, cuts="
          f"({cuts[0]:.0f}, {cuts[1]:.0f}, {cuts[2]:.0f}), "
          f"score C={c_fold:.3f}")

folds = pd.DataFrame(fold_rows)
folds.to_csv("score_validation_by_fold.csv", index=False)

print("Age bracket stability across folds:")
for c in ["age_cut1", "age_cut2", "age_cut3"]:
    print(f"  {c}: mean {folds[c].mean():.1f}, "
          f"range {folds[c].min():.0f}-{folds[c].max():.0f}")

# Cross-validated score performance
c_cv, lo_cv, hi_cv = bootstrap_cindex(df["event"], df["time"], oof_score,
                                      n_boot=N_BOOT, seed=SEED)
print(f"\nCross-validated score C = "
      f"{cindex(df['event'], df['time'], oof_score):.4f} "
      f"[{lo_cv:.3f}, {hi_cv:.3f}]")

# The published score, applied to the same patients it was derived from
df["score_published"] = integer_score(df, age_cuts=PUBLISHED_AGE_CUTS)
c_ap = cindex(df["event"], df["time"], df["score_published"])
_, lo_ap, hi_ap = bootstrap_cindex(df["event"], df["time"],
                                   df["score_published"],
                                   n_boot=N_BOOT, seed=SEED)
print(f"Apparent score C = {c_ap:.4f} [{lo_ap:.3f}, {hi_ap:.3f}]")
print(f"Optimism = {c_ap - cindex(df['event'], df['time'], oof_score):+.4f}")

df["score_cv"] = oof_score
df["tier_cv"] = score_tier(oof_score)

rows = []
for t, lbl in [("Low", "0-1"), ("Medium", "2-3"), ("High", "4+")]:
    m = df["tier_cv"] == t
    if m.sum() == 0:
        continue
    r, lo, hi = annual_rate(df.loc[m, "time"], df.loc[m, "event"])
    rows.append({"Tier": t, "Score_range": lbl, "N": int(m.sum()),
                 "Events": int(df.loc[m, "event"].sum()),
                 "rate_pct": round(100 * r, 2),
                 "lo_pct": round(100 * lo, 2), "hi_pct": round(100 * hi, 2)})
tiers = pd.DataFrame(rows)
tiers.to_csv("score_validation_tiers.csv", index=False)
print("\nTiers from the cross-validated score:")
print(tiers.to_string(index=False))
lr = multivariate_logrank_test(df["time"], df["tier_cv"], df["event"])
print(f"Grouped log-rank chi2 {lr.test_statistic:.1f}, p {lr.p_value:.3g}")
pd.DataFrame([
    {"metric": "score_C_crossvalidated",
     "value": round(cindex(df["event"], df["time"], oof_score), 4)},
    {"metric": "score_C_cv_lo", "value": round(lo_cv, 4)},
    {"metric": "score_C_cv_hi", "value": round(hi_cv, 4)},
    {"metric": "score_C_apparent", "value": round(c_ap, 4)},
    {"metric": "optimism",
     "value": round(c_ap - cindex(df["event"], df["time"], oof_score), 4)},
    {"metric": "logrank_chi2", "value": round(float(lr.test_statistic), 2)},
    {"metric": "logrank_p", "value": float(lr.p_value)},
]).to_csv("score_validation_summary.csv", index=False)

df[["time", "event", "score_published", "score_cv", "tier_cv"]].to_csv(
    "score_validation_predictions.csv", index=False)

import glob
print(f"\n{len(glob.glob('*.csv'))} files in {RESULTS_DIR}")