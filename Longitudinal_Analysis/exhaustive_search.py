"""Exhaustive feature-subset search used to choose the model's predictors (separate from main.py).

Fits every subset of the candidate pool with a ridge Cox model, ranks them by mean time-dependent AUC,
then re-scores the top subsets under nested cross-validation and compares with the final 3 predictors.

Usage: python exhaustive_search.py [path/to/registry.xlsx]

Writes to bavm_exhaustive_search_<date>/ next to the data file:
  exhaustive_screening_all.csv      every subset with its screening AUC and C-index
  exhaustive_plateau_by_size.csv    best subset at each size
  exhaustive_top_nested.csv         top subsets re-scored under nested CV
  exhaustive_vs_published.csv       final 3-predictor model vs the exhaustive winner
"""
import argparse
import glob
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

import config as cfg
from bavm.data import load_and_engineer, surv_array
from bavm.stats import build_pipe, cindex, nested_cv_risk

MIN_SIZE = 2
MAX_SIZE = 12
SCREEN_ALPHA = 1.0        # single alpha for the fast ranking pass
SCREEN_FOLDS = 3
TOP_N = 50                # subsets re-scored under nested CV
N_JOBS = -1
AUC_TIMES = np.array([1.0, 2.0, 3.0, 5.0], dtype=float)


def screen(df, y, ev, feats):
    """Fast pass: 3-fold ridge Cox, mean td-AUC and C-index."""
    X = df[list(feats)].astype(float)
    skf = StratifiedKFold(n_splits=SCREEN_FOLDS, shuffle=True, random_state=cfg.SEED)
    risk = np.full(len(X), np.nan)
    empty = {"features": feats, "mean_auc": np.nan, "c": np.nan}
    for tr, ts in skf.split(X, ev):
        try:
            risk[ts] = build_pipe(SCREEN_ALPHA).fit(X.iloc[tr], y[tr]).predict(X.iloc[ts])
        except Exception:
            return empty
    if np.isnan(risk).any() or np.std(risk) < 1e-10:
        return empty
    try:
        t = AUC_TIMES[AUC_TIMES < df["time"].max()]
        _, mean_auc = cumulative_dynamic_auc(y, y, risk, t)
    except Exception:
        return empty
    return {"features": feats, "mean_auc": float(mean_auc),
            "c": cindex(df["event"], df["time"], risk)}


def main(data_path):
    np.random.seed(cfg.SEED)
    out = os.path.join(os.path.dirname(os.path.abspath(data_path)),
                       f"bavm_exhaustive_search_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out, exist_ok=True)
    os.chdir(out)

    df = load_and_engineer(data_path)
    y = surv_array(df)
    ev = df["event"].to_numpy().astype(int)
    pool = [f for f in cfg.CANDIDATE_FEATURES if f in df.columns]
    print(f"Candidate pool ({len(pool)}): {pool}")
    subsets = [c for k in range(MIN_SIZE, min(MAX_SIZE, len(pool)) + 1)
               for c in combinations(pool, k)]
    print(f"Subsets of size {MIN_SIZE}-{MAX_SIZE}: {len(subsets)}")

    # Screening
    scr = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(screen)(df, y, ev, f) for f in subsets)).dropna(subset=["mean_auc"])
    scr["k"] = scr["features"].apply(len)
    scr["features"] = scr["features"].apply(lambda t: ", ".join(t))
    scr = scr.sort_values("mean_auc", ascending=False).reset_index(drop=True)
    scr.to_csv("exhaustive_screening_all.csv", index=False)
    print(f"Ranked {len(scr)} subsets, saved exhaustive_screening_all.csv")

    plateau = scr.loc[scr.groupby("k")["mean_auc"].idxmax()].sort_values("k")
    plateau.to_csv("exhaustive_plateau_by_size.csv", index=False)
    print("\nBest subset at each size:")
    print(plateau[["k", "mean_auc", "c", "features"]].to_string(index=False))

    # Top subsets under nested CV
    rows = []
    for _, r in scr.head(TOP_N).iterrows():
        feats = [f.strip() for f in r["features"].split(",")]
        try:
            risk, fold_c, _ = nested_cv_risk(df[feats].astype(float), y, ev, cfg.OUTER_FOLDS,
                                             cfg.INNER_FOLDS, cfg.ALPHA_GRID, cfg.SEED)
            rows.append({"features": r["features"], "k": len(feats),
                         "screen_auc": round(r["mean_auc"], 4),
                         "nested_C": round(cindex(df["event"], df["time"], risk), 4),
                         "fold_C_sd": round(float(np.nanstd(fold_c, ddof=1)), 4)})
        except Exception as e:
            print(f"  {r['features']}: {type(e).__name__}")
    resc = pd.DataFrame(rows).sort_values("nested_C", ascending=False)
    resc.to_csv("exhaustive_top_nested.csv", index=False)
    print(resc.head(15).to_string(index=False))

    # Final 3-predictor model for comparison
    base_risk, _, _ = nested_cv_risk(df[cfg.BASE].astype(float), y, ev, cfg.OUTER_FOLDS,
                                     cfg.INNER_FOLDS, cfg.ALPHA_GRID, cfg.SEED)
    base_c = cindex(df["event"], df["time"], base_risk)
    best = resc.iloc[0]
    print(f"\nFinal 3-predictor model: nested C = {base_c:.4f}")
    print(f"Exhaustive winner:       nested C = {best['nested_C']:.4f} ({best['features']})")
    print(f"Difference:              {best['nested_C'] - base_c:+.4f}")
    pd.DataFrame([
        {"model": "3-feature Cox", "features": ", ".join(cfg.BASE), "nested_C": round(base_c, 4)},
        {"model": "exhaustive winner", "features": best["features"], "nested_C": best["nested_C"]},
        {"model": "difference", "features": "-", "nested_C": round(best["nested_C"] - base_c, 4)},
    ]).to_csv("exhaustive_vs_published.csv", index=False)
    print(f"\n{len(glob.glob('*.csv'))} files in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=cfg.DATA_PATH)
    main(ap.parse_args().data)