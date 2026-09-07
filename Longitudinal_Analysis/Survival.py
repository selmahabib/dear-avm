"""
Longitudinal Analysis
OUTPUTS

  Figures  FIGURE_longitudinal.png       
           km_untreated.png
           score_distribution.png

  Tables   cox_univariate.csv            
           cox_multivariable.csv         
           cox_multivariable_vif.csv
           cox_multivariable_schoenfeld.csv
           ridge_cox_summary.csv
           risk_tiers.csv
           calibration_by_quartile.csv
           predicted_risk_profiles.csv   
           untreated_sensitivity.csv
           cohort_summary.csv
           score_derivation.csv
           rate_by_score.csv
           cv_predictions.csv
"""
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import linregress
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from survival_data import (
    CANDIDATE_FEATURES, FINAL_FEATURES, R2ED_FEATS,
    cohort_summary, load_and_engineer, surv_array,
)
from survival_helpers import (
    ALPHA_GRID, annual_rate, bootstrap_cindex, build_pipe, cindex,
    derive_integer_score, followup_summary, integer_score, km_observed,
    nested_cv_risk, score_tier,
)
from cox_analysis import multivariable_cox, univariate_cox
from generate_figures import (
    figure_km_untreated, figure_longitudinal, figure_score_distribution,
)

# Settings
DATA_PATH = r'path/to/your/data.xlsx'
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 1000
CALIB_HORIZON = 2.0
CALIB_QUANTILES = 4
MIN_CALIB_BIN = 5       # bins smaller than this are dropped
CALIB_AXIS_CAP = 0.14   
SCREEN_P = 0.10
TD_AUC_TIMES = np.array([0.5, 1, 1.5, 2, 3, 5, 7, 10], dtype=float)

np.random.seed(SEED)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Registry not found - check DATA_PATH at the top of this file.")

_src = os.path.dirname(os.path.abspath(DATA_PATH))
RESULTS_DIR = os.path.join(
    _src, f"bavm_longitudinal_analysis_{datetime.now():%Y%m%d_%H%M}")
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '.last_run'), 'w') as fh:
    fh.write(RESULTS_DIR)
os.chdir(RESULTS_DIR)

USE_DERIVED_CUTS = False   # True to use this cohort's derived age brackets
PUBLISHED_AGE_CUTS = (27, 45, 63)

# Data
df = load_and_engineer(DATA_PATH)

# cohort_summary prints and saves the per-group table
print("\nCOHORT:")
cohort_summary(df)
fu = followup_summary(df)
y = surv_array(df)
ev = df["event"].to_numpy().astype(int)

# Univariate screening
print("\nUNIVARIATE COX:")
uni = univariate_cox(df, CANDIDATE_FEATURES)
screened = uni.loc[uni["p"] < SCREEN_P, "Feature"].tolist()
print(f"Passed p<{SCREEN_P}: {screened}")

# Multivariable, VIF, proportional hazards
print("\nMULTIVARIABLE COX:")
multi, vif, schoenfeld = multivariable_cox(df, screened)
print(multi.to_string(index=False))
print(f"Max VIF {vif['VIF'].max():.2f}")

# Final ridge Cox on the retained features
print(f"\nRIDGE COX ({OUTER_FOLDS} outer x {INNER_FOLDS} inner):")
X = df[FINAL_FEATURES].astype(float)
oof_risk, fold_c, alphas = nested_cv_risk(
    X, y, ev, n_outer=OUTER_FOLDS, n_inner=INNER_FOLDS,
    alphas=ALPHA_GRID, seed=SEED)
FINAL_ALPHA = float(np.median(alphas))
_, _, _, oof_prob = nested_cv_risk(
    X, y, ev, n_outer=OUTER_FOLDS, n_inner=INNER_FOLDS,
    alphas=ALPHA_GRID, seed=SEED, horizon=CALIB_HORIZON,
    fixed_alpha=FINAL_ALPHA)

c_mean, c_lo, c_hi = bootstrap_cindex(df["event"], df["time"], oof_risk,
                                      n_boot=N_BOOT, seed=SEED)
print(f"C-index {cindex(df['event'], df['time'], oof_risk):.4f} "
      f"[{c_lo:.3f}, {c_hi:.3f}] | per-fold {np.nanmean(fold_c):.3f} "
      f"(SD {np.nanstd(fold_c, ddof=1):.3f}) | alphas {alphas}")

# Full-cohort refit for the reported coefficients and baseline survival
final_pipe = build_pipe(FINAL_ALPHA).fit(X, y)
model = final_pipe.named_steps["model"]
coefs = pd.DataFrame({"Feature": FINAL_FEATURES,
                      "beta": np.round(model.coef_, 4),
                      "HR": np.round(np.exp(model.coef_), 3)})
coefs.to_csv("ridge_cox_summary.csv", index=False)
print(coefs.to_string(index=False))

_pred = pd.DataFrame({"time": df["time"], "event": df["event"],
                      "cv_risk": oof_risk})
for _c in ["Race_nonwhite", "Gender", "Ever_Ruptured"]:
    if _c in df.columns:
        _pred[_c] = df[_c].to_numpy()
_pred.to_csv("cv_predictions.csv", index=False)

# R2eD baseline 
r2ed_feats = [f for f in R2ED_FEATS if f in df.columns]
r2ed_risk, r2ed_fold_c, _ = nested_cv_risk(
    df[r2ed_feats].astype(float), y, ev,
    n_outer=OUTER_FOLDS, n_inner=INNER_FOLDS, alphas=ALPHA_GRID, seed=SEED)
print(f"R2eD C-index {cindex(df['event'], df['time'], r2ed_risk):.4f}")

# Integer score derived from the fitted coefficients
print("\nSCORE DERIVATION:")
_age = pd.to_numeric(df["Age_dx"], errors="coerce")
derived_cuts, score_table = derive_integer_score(
    model.coef_, FINAL_FEATURES,
    age_mean=float(_age.mean()), age_sd=float(_age.std(ddof=1)))
age_cuts = tuple(derived_cuts) if USE_DERIVED_CUTS else PUBLISHED_AGE_CUTS
print(f"Using age cuts {tuple(round(c) for c in age_cuts)}"
      f"{' (derived)' if USE_DERIVED_CUTS else ' (published)'}; "
      f"derived were {tuple(round(c) for c in derived_cuts)}")

df["score"] = integer_score(df, age_cuts=age_cuts)
df["tier"] = score_tier(df["score"].to_numpy())

_max_score = int(df["score"].max())
rate_rows = []
for t, lbl in [("Low", "0-1"), ("Medium", "2-3"),
               ("High", f"4-{_max_score}")]:
    m = df["tier"] == t
    r, lo, hi = annual_rate(df.loc[m, "time"], df.loc[m, "event"])
    rate_rows.append({"Tier": t, "Score_range": lbl, "N": int(m.sum()),
                      "Events": int(df.loc[m, "event"].sum()),
                      "rate_pct": 100 * r, "lo_pct": 100 * lo,
                      "hi_pct": 100 * hi})
rate_df = pd.DataFrame(rate_rows)
rate_df.round(3).to_csv("risk_tiers.csv", index=False)
print("\nRISK TIERS:")
print(rate_df.round(2).to_string(index=False))

# Rate at each individual score level before collapsing into tiers
score_rows = []
for sc in sorted(df["score"].dropna().unique()):
    m = df["score"] == sc
    r, lo, hi = annual_rate(df.loc[m, "time"], df.loc[m, "event"])
    score_rows.append({"Score": int(sc), "N": int(m.sum()),
                       "Events": int(df.loc[m, "event"].sum()),
                       "rate_pct": 100 * r, "lo_pct": 100 * lo,
                       "hi_pct": 100 * hi})
score_rate_df = pd.DataFrame(score_rows)
score_rate_df.round(3).to_csv("rate_by_score.csv", index=False)
print("\nRATE BY SCORE:")
print(score_rate_df.round(2).to_string(index=False))
figure_score_distribution(df["score"].to_numpy(), score_rate_df)
print("Saved score_distribution.png")

lr = multivariate_logrank_test(df["time"], df["tier"], df["event"])
print(f"Grouped log-rank chi2 {lr.test_statistic:.1f}, p {lr.p_value:.3g}")
S0 = final_pipe.named_steps["model"].baseline_survival_
S0_x, S0_y = np.asarray(S0.x, float), np.asarray(S0.y, float)
s0_t = float(np.interp(CALIB_HORIZON, S0_x, S0_y))
# Out-of-fold predicted probabilities each from its own fold's baseline.
df["pred_event_prob"] = oof_prob
df["_q"] = pd.qcut(df["pred_event_prob"], CALIB_QUANTILES,
                   labels=False, duplicates="drop")
calib_rows = []
for q, g in df.groupby("_q"):
    if len(g) < MIN_CALIB_BIN:
        continue
    obs, lo, hi = km_observed(g["time"], g["event"], CALIB_HORIZON)
    calib_rows.append({"quartile": int(q) + 1, "n": len(g),
                       "pred": g["pred_event_prob"].mean(),
                       "obs": obs, "obs_lo": lo, "obs_hi": hi})
calib_df = pd.DataFrame(calib_rows)
calib_df.round(4).to_csv("calibration_by_quartile.csv", index=False)
slope, intercept, _, _, _ = linregress(calib_df["pred"], calib_df["obs"])
slope, intercept = float(slope), float(intercept)
print(f"\nCALIBRATION at {CALIB_HORIZON:.0f}yr: slope {slope:.3f}, "
      f"intercept {intercept:.4f}")
print(calib_df.round(4).to_string(index=False))

# Time-dependent AUC
tmax = float(np.percentile(df.loc[df["event"] == 1, "time"], 95))
times = TD_AUC_TIMES[TD_AUC_TIMES < min(tmax, df["time"].max())]
auc_model, _ = cumulative_dynamic_auc(y, y, oof_risk, times)
auc_r2ed, _ = cumulative_dynamic_auc(y, y, r2ed_risk, times)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(200):
    idx = rng.choice(len(df), len(df), replace=True)
    if df["event"].to_numpy()[idx].sum() < 5:
        continue
    try:
        a, _ = cumulative_dynamic_auc(y, y[idx], oof_risk[idx], times)
        boot.append(a)
    except Exception:
        continue
if boot:
    boot = np.vstack(boot)
    auc_lo = np.nanpercentile(boot, 2.5, axis=0)
    auc_hi = np.nanpercentile(boot, 97.5, axis=0)
else:
    auc_lo = auc_hi = None

rng = np.random.default_rng(SEED + 1)
boot_r = []
for _ in range(200):
    idx = rng.choice(len(df), len(df), replace=True)
    if df["event"].to_numpy()[idx].sum() < 5:
        continue
    try:
        a, _ = cumulative_dynamic_auc(y, y[idx], r2ed_risk[idx], times)
        boot_r.append(a)
    except Exception:
        continue
if boot_r:
    boot_r = np.vstack(boot_r)
    r2ed_lo = np.nanpercentile(boot_r, 2.5, axis=0)
    r2ed_hi = np.nanpercentile(boot_r, 97.5, axis=0)
else:
    r2ed_lo = r2ed_hi = None
print(f"\ntd-AUC: " + ", ".join(f"{t:.0f}yr {a:.3f}"
                                for t, a in zip(times, auc_model)))

# Figure 3
figure_longitudinal(df, df["tier"].to_numpy(), lr.p_value,
                    cindex(df["event"], df["time"], oof_risk), c_lo, c_hi,
                    rate_df, calib_df, slope, intercept, CALIB_HORIZON,
                    times, auc_model, auc_lo, auc_hi, auc_r2ed,
                    r2ed_lo=r2ed_lo, r2ed_hi=r2ed_hi,
                    calib_cap=CALIB_AXIS_CAP,
                    fu_mean=fu["Followup_mean_yr"],
                    fu_sd=float(df["time"].std()),
                    fu_median=fu["Followup_median_yr"],
                    fu_iqr=fu["Followup_IQR"])
print("Saved FIGURE_longitudinal.png and .pdf")

# Predicted risk profiles
profiles = []
for age, deep, prior in [(17, 0, 0), (36, 0, 0), (36, 0, 1), (36, 1, 0),
                         (36, 1, 1), (55, 1, 0), (55, 1, 1), (73, 1, 1)]:
    row = pd.DataFrame([[age, deep, prior]], columns=FINAL_FEATURES)
    lp = float(final_pipe.predict(row)[0]) - float(np.nanmean(oof_risk))
    s = integer_score(row, age_cuts=age_cuts).iloc[0]
    r2 = 1 - float(np.interp(2.0, S0_x, S0_y)) ** np.exp(lp)
    r5 = 1 - float(np.interp(5.0, S0_x, S0_y)) ** np.exp(lp)
    profiles.append({"Score": int(s), "Age": age, "Deep": bool(deep),
                     "Prior_rupture": bool(prior),
                     "risk_2yr_pct": round(100 * r2, 1),
                     "risk_5yr_pct": round(100 * r5, 1)})
prof_df = pd.DataFrame(profiles)
prof_df.to_csv("predicted_risk_profiles.csv", index=False)
print("\nPREDICTED RISK PROFILES:")
print(prof_df.to_string(index=False))

# Untreated sensitivity: coefficients applied without re-estimation
unt = df[df["untreated"]].copy()
if len(unt) > 20 and unt["event"].sum() >= 5:
    risk_unt = final_pipe.predict(unt[FINAL_FEATURES].astype(float))
    c_u, lo_u, hi_u = bootstrap_cindex(unt["event"], unt["time"], risk_unt,
                                       n_boot=N_BOOT, seed=SEED)
    c_s, lo_s, hi_s = bootstrap_cindex(unt["event"], unt["time"],
                                       unt["score"], n_boot=N_BOOT, seed=SEED)
    y_unt = surv_array(unt)
    t_unt = times[times < unt["time"].max()]
    auc_unt, _ = cumulative_dynamic_auc(y, y_unt, risk_unt, t_unt)
    pd.DataFrame({"metric": ["N", "events",
                             "C_linear_predictor", "C_lp_lo", "C_lp_hi",
                             "C_integer_score", "C_score_lo", "C_score_hi"]
                            + [f"tdAUC_{t:.0f}yr" for t in t_unt],
                  "value": [len(unt), int(unt["event"].sum()),
                            round(c_u, 4), round(lo_u, 4), round(hi_u, 4),
                            round(c_s, 4), round(lo_s, 4), round(hi_s, 4)]
                           + [round(float(a), 4) for a in auc_unt]}
                 ).to_csv("untreated_sensitivity.csv", index=False)
    print(f"\nUNTREATED SUBSET: n={len(unt)}, events={int(unt['event'].sum())}")
    print(f"  linear predictor C={c_u:.3f} [{lo_u:.3f}, {hi_u:.3f}]")
    print(f"  integer score    C={c_s:.3f} [{lo_s:.3f}, {hi_s:.3f}]")

    figure_km_untreated(unt, score_tier(unt["score"].to_numpy()))
    print("Saved km_untreated.png")

import glob
print(f"\n{len(glob.glob('*.png')) + len(glob.glob('*.csv'))} files in {RESULTS_DIR}")