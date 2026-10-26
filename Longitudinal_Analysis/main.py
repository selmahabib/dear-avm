"""Fits and validates the bAVM rupture model with a time-varying prior-rupture effect.

Usage: python main.py [path/to/registry.xlsx] [--out results_dir]

Writes to results_<date>/:
  FIGURE_prior_rupture_time_varying(.png/.pdf, _single)  how the prior-rupture HR changes over time
  FIGURE_prior_vs_no_prior_risk                          cumulative incidence and interval rates
  FIGURE_tv_model                                        hazard ratios and calibration of the time-varying model
  FIGURE_longitudinal_time_varying                       main figure: td-AUC, calibration, KM and incidence by tier
  FIGURE_tv_calibration_horizons, FIGURE_tv_score_calibration_horizons (+ one png per horizon)
  tv_prior_*.csv, prior_vs_no_prior_rates.csv            exploratory tables
  tv_model_*.csv, table4_risk_profiles_tv.csv            coefficients, folds, C-index, calibration, points, profiles
  tv_risk_tiers.csv, tv_td_auc*.csv, tv_cindex_summary.csv, tv_fairness_subgroups.csv
  tv_score_variants.csv, tv_penalty_grid_sensitivity.csv, calibration_*_2yr.csv
  supplement/                                            S-figures and S-tables (calibration at 2/5/10 yr, observed vs predicted)
"""
import argparse
import os
import shutil
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

import config as cfg
from bavm import plotting as P
from bavm.calibration import (calib_table, calibration_measures, cox_cal_slope,
                              quartile_fit, score_calib_table, smooth_calibration)
from bavm.data import load_and_engineer, surv_array
from bavm.episodes import fit_ctv, split
from bavm.evaluate import boot_auc, boot_c, interval_rates, logrank_counting
from bavm.explore import run_time_varying_prior
from bavm.model import (cindex_tv, derive_points, fit_model, fit_score_model,
                        inner_shrink, lp_pair, points_from, predict_generic,
                        predict_risk, score_with, select_cut, tune_pen, tv_scores)
from bavm.stats import annual_rate, km_observed, nested_cv_risk, score_tier

AGE, DEEP, PRIOR, BASE = cfg.AGE, cfg.DEEP, cfg.PRIOR, cfg.BASE


def header(text):
    print(f"\n{text}\n{'-' * len(text)}")


def main(data_path, out_dir):
    np.random.seed(cfg.SEED)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    print(f"Results -> {out_dir}")
    primary = (cfg.SCORE_POINTS_FROM, cfg.SCORE_RECALIBRATE)
    H = cfg.CALIB_HORIZON

    # Data
    df = load_and_engineer(data_path)
    keep = df[["time", "event"] + BASE].notna().all(axis=1).to_numpy()
    df_k = df.loc[keep].reset_index(drop=True)
    d = df_k[["time", "event"] + BASE].astype(float).reset_index(drop=True)
    d["time"] = d["time"].clip(lower=cfg.MIN_TIME)
    T, E = d["time"].to_numpy(), d["event"].to_numpy().astype(int)
    n = len(d)
    print(f"Cohort: n={n}, events={int(E.sum())}, predictors={BASE}")

    # Stage 1: when does the prior-rupture HR change?
    header("STAGE 1: WHEN DOES THE PRIOR-RUPTURE HR CHANGE?")
    TV = run_time_varying_prior(d, BASE, prior_col=PRIOR, piecewise_cuts=cfg.PIECEWISE_CUTS)

    groups = [(1, "Prior rupture (re-hemorrhage)", P.HOPKINS_RED),
              (0, "No prior rupture (first hemorrhage)", P.NAVY)]
    rt, labels = interval_rates(d, groups, cfg.PIECEWISE_CUTS)
    rt.round(3).to_csv("prior_vs_no_prior_rates.csv", index=False)
    P.figure_prior_vs_no_prior(d, rt, labels, groups,
                               dict(zip(TV["piecewise"]["Interval"], TV["piecewise"]["HR"])),
                               cfg.PIECEWISE_CUTS, PRIOR)
    print("\nRUPTURE RATE BY INTERVAL, prior vs no prior rupture (per 100 person-years):")
    print(rt[["Group", "Interval", "Events", "Person_years", "rate_pct", "lo_pct",
              "hi_pct"]].round(2).to_string(index=False))

    # Stage 2: nested CV
    header(f"STAGE 2: NESTED CV ({cfg.OUTER_FOLDS} outer x {cfg.INNER_FOLDS} inner)")
    tmax = float(np.percentile(T[E == 1], 95))
    td_times = cfg.TD_AUC_TIMES[cfg.TD_AUC_TIMES < min(tmax, T.max())]
    oof_h = {"tv": {h: np.full(n, np.nan) for h in cfg.CALIB_HORIZONS}}
    sv = {v: {"se": np.full(n, np.nan), "sl": np.full(n, np.nan),
              "lp_e": np.full(n, np.nan), "lp_l": np.full(n, np.nan),
              "risk": {h: np.full(n, np.nan) for h in cfg.CALIB_HORIZONS},
              "pts": [], "shrink": []} for v in cfg.SCORE_VARIANTS}
    td_tv = np.full((n, len(td_times)), np.nan)
    oof = {k: np.full(n, np.nan) for k in
           ["tv_le", "tv_ll", "tv_cut", "tv_risk", "fold"]}
    fold_rows = []
    outer = StratifiedKFold(cfg.OUTER_FOLDS, shuffle=True, random_state=cfg.SEED)
    for f, (tr, te) in enumerate(outer.split(d, d["event"])):
        train = d.iloc[tr].reset_index(drop=True)
        test = d.iloc[te].reset_index(drop=True)
        cut = select_cut(train)
        p_tv = tune_pen(train, cut, cfg.SEED + f)
        m_tv = fit_model(train, cut, p_tv)

        le, ll = lp_pair(test, m_tv)
        oof["tv_le"][te], oof["tv_ll"][te], oof["tv_cut"][te] = le, ll, cut
        oof["tv_risk"][te] = predict_risk(test, m_tv, H)
        oof["fold"][te] = f + 1
        for h in cfg.CALIB_HORIZONS:
            oof_h["tv"][h][te] = predict_risk(test, m_tv, h)

        # Integer score derived on the training fold
        for var in cfg.SCORE_VARIANTS:
            pen_v = p_tv if var[0] == "penalized" else 0.0
            pts_f = derive_points(train, cut, pen_v)
            se_tr, sl_tr = tv_scores(train, pts_f)
            se_te, sl_te = tv_scores(test, pts_f)
            shrink = inner_shrink(train, cut, pen_v, cfg.SEED + f) if var[1] else 1.0
            m_s = fit_score_model(train, se_tr, sl_tr, cut, shrink=shrink)
            V = sv[var]
            V["se"][te], V["sl"][te] = se_te, sl_te
            V["lp_e"][te], V["lp_l"][te] = m_s["b"] * se_te, m_s["b"] * sl_te
            for h in cfg.CALIB_HORIZONS:
                V["risk"][h][te] = predict_generic(m_s["b"] * se_te, m_s["b"] * sl_te,
                                                   m_s["tk"], m_s["dH"], cut, h)
            V["pts"].append(pts_f)
            V["shrink"].append(round(shrink, 3))
        for k, t in enumerate(td_times):
            td_tv[te, k] = predict_risk(test, m_tv, t)

        c_tv = cindex_tv(test["time"], test["event"], le, ll, cut)
        fold_rows.append({"Fold": f + 1, "cut_yr": cut, "pen_tv": p_tv,
                          "HR_early": np.exp(m_tv["b_early"]), "HR_late": np.exp(m_tv["b_late"]),
                          "C_time_varying": c_tv,
                          "events_test": int(test["event"].sum())})
        print(f"  Fold {f + 1}: cut={cut:g} yr, penalizer={p_tv:g} | "
              f"HR early {np.exp(m_tv['b_early']):.2f} late {np.exp(m_tv['b_late']):.2f} | "
              f"C {c_tv:.3f}")

    folds = pd.DataFrame(fold_rows)
    folds.round(4).to_csv("tv_model_folds.csv", index=False)
    SP = sv[primary]
    oof_h["score"] = SP["risk"]
    print(f"Integer score variant used: points from {cfg.SCORE_POINTS_FROM} model, "
          f"recalibration {'on' if cfg.SCORE_RECALIBRATE else 'off'} (shrink per fold {SP['shrink']})")

    C_TV, c_lo, c_hi, _, _ = boot_c(T, E, oof["tv_le"], oof["tv_ll"], oof["tv_cut"])
    cidx = pd.DataFrame([{"Model": "Time-varying prior rupture", "C": C_TV, "lo": c_lo, "hi": c_hi}])
    cidx.round(4).to_csv("tv_model_cindex.csv", index=False)
    print("\nCROSS-VALIDATED C-INDEX:")
    print(cidx.round(3).to_string(index=False))

    # Full-cohort final models
    CUT = cfg.FIXED_CUT if cfg.FIXED_CUT is not None else select_cut(d)
    PEN_TV = float(np.median(folds["pen_tv"]))
    how = "pre-specified" if cfg.FIXED_CUT is not None else "profile likelihood, full cohort"
    print(f"\nFINAL MODEL: cut-point = {CUT:g} yr ({how}), "
          f"penalizer = {PEN_TV:g}")

    u_tv = fit_model(d, CUT, 0.0, scale=False)
    lab = {"age": f"{AGE} (per year)", "deep": "Exclusive deep location",
           "prior_early": f"Prior rupture, 0-{CUT:g} yr",
           "prior_late": f"Prior rupture, >{CUT:g} yr"}
    coef_rows = []
    s = u_tv["ctv"].summary
    for c in u_tv["cols"]:
        coef_rows.append({"Term": lab[c], "beta": s.loc[c, "coef"],
                          "HR": s.loc[c, "exp(coef)"],
                          "HR_lo": s.loc[c, "exp(coef) lower 95%"],
                          "HR_hi": s.loc[c, "exp(coef) upper 95%"],
                          "p": s.loc[c, "p"]})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.round(4).to_csv("tv_model_coefficients.csv", index=False)
    print("\nHAZARD RATIOS (unpenalized, full cohort):")
    print(coef_df.round(3).to_string(index=False))

    # Wald test early vs late
    names = list(u_tv["ctv"].params_.index)
    Vm = np.asarray(u_tv["ctv"].variance_matrix_, float)
    ie, il = names.index("prior_early"), names.index("prior_late")
    z = (u_tv["b_early"] - u_tv["b_late"]) / np.sqrt(Vm[ie, ie] + Vm[il, il] - 2 * Vm[ie, il])
    P_WALD = 2 * norm.sf(abs(z))
    print(f"  Wald test early vs late prior-rupture HR: z = {z:.2f}, p = {P_WALD:.4f}")

    f_tv = fit_model(d, CUT, PEN_TV)
    pd.DataFrame([
        {"penalizer": PEN_TV, "b_age": f_tv["b_age"],
         "b_deep": f_tv["b_deep"], "b_prior_early": f_tv["b_early"],
         "b_prior_late": f_tv["b_late"], "cut_yr": CUT}]
    ).round(5).to_csv("tv_model_penalized_coefs.csv", index=False)
    for h in (1, 2, 5, 10):
        print(f"H0({h}) = {f_tv['dH'][f_tv['tk'] <= h].sum():.5f}")

    # Integer score points
    src_tv = f_tv if cfg.SCORE_POINTS_FROM == "penalized" else u_tv
    PD_tv, PE_tv, PL_tv = points_from(src_tv)
    s_e, s_l = score_with(d, PD_tv, PE_tv), score_with(d, PD_tv, PL_tv)
    C_STV = cindex_tv(T, E, s_e, s_l, CUT)
    pts = pd.DataFrame([
        {"Score": "Time-varying", "Age": "0/1/2/3 by MARS bracket", "Deep_location": PD_tv,
         "Prior_rupture_early": PE_tv, "Prior_rupture_late": PL_tv, "cut_yr": CUT,
         "C_apparent": C_STV}])
    pts.round(4).to_csv("tv_model_score_points.csv", index=False)
    print(f"\nINTEGER SCORE POINTS ({cfg.SCORE_POINTS_FROM} betas; apparent C-index):")
    print(pts.round(3).to_string(index=False))

    # Predicted risk profiles
    prof = pd.DataFrame(cfg.PROFILES, columns=[AGE, DEEP, PRIOR]).astype(float)
    prof_rows = []
    for h in (H, 5.0):
        rtv = predict_risk(prof, u_tv, h)
        for i, (a, dp, pr) in enumerate(cfg.PROFILES):
            prof_rows.append({"Age": a, "Deep": bool(dp), "Prior_rupture": bool(pr),
                              "horizon_yr": h, "risk_time_varying_pct": 100 * rtv[i]})
    prof_df = (pd.DataFrame(prof_rows)
               .pivot_table(index=["Age", "Deep", "Prior_rupture"], columns="horizon_yr",
                            values=["risk_time_varying_pct"])
               .round(1))
    prof_df.columns = [f"{a}_{b:g}yr" for a, b in prof_df.columns]
    prof_df = prof_df.reset_index()
    prof_df.to_csv("tv_model_risk_profiles.csv", index=False)
    print("\nPREDICTED RISK PROFILES (%):")
    print(prof_df.to_string(index=False))

    # Table 4: final penalized model
    sc_early = score_with(prof, PD_tv, PE_tv).astype(int)
    sc_late = score_with(prof, PD_tv, PL_tv).astype(int)
    t4_risk = {h: predict_risk(prof, f_tv, h) for h in cfg.TABLE4_HORIZONS}
    t4_rows = []
    for i, (a, dp, pr) in enumerate(cfg.PROFILES):
        row = {"Score_year1": sc_early[i], "Score_after_year1": sc_late[i], "Age": a,
               "Deep": "Yes" if dp else "No", "Prior_rupture": "Yes" if pr else "No"}
        for h in cfg.TABLE4_HORIZONS:
            row[f"Risk_{h:g}yr_pct"] = round(100 * t4_risk[h][i], 1)
        t4_rows.append(row)
    table4 = pd.DataFrame(t4_rows).sort_values(["Score_year1", "Age"]).reset_index(drop=True)
    table4.to_csv("table4_risk_profiles_tv.csv", index=False)
    print(f"\nTABLE 4 (final penalized TV model, penalizer={PEN_TV:g}, cut={CUT:g} yr; "
          f"points: deep={PD_tv}, prior early={PE_tv}, prior late={PL_tv}):")
    print(table4.to_string(index=False))

    # Calibration at the main horizon, both models
    cal = calib_table(d, oof["tv_risk"], H)
    cal.round(4).to_csv("tv_model_calibration.csv", index=False)
    print(f"\nCALIBRATION at {H:g} yr (out-of-fold):")
    print(cal.round(4).to_string(index=False))

    pd.DataFrame({"time": T, "event": E, "fold": oof["fold"],
                  "tv_lp_early": oof["tv_le"], "tv_lp_late": oof["tv_ll"],
                  "tv_cut_yr": oof["tv_cut"], "tv_risk_2yr": oof["tv_risk"]}
                 ).to_csv("tv_model_cv_predictions.csv", index=False)

    P.figure_tv_model(u_tv, cal, P_WALD, CUT, H)
    print("\nSaved FIGURE_tv_model.png / .pdf")

    # Calibration slopes
    header("MANUSCRIPT FIGURES (time-varying model)")
    TV_SLOPE = cox_cal_slope(d, oof["tv_le"], oof["tv_ll"], oof["tv_cut"])
    print(f"Calibration slope (Cox on OOF LP): {TV_SLOPE[0]:.3f} "
          f"({TV_SLOPE[1]:.3f}-{TV_SLOPE[2]:.3f})")

    # Time-updated score tiers
    ep_t = split(d.assign(_se=s_e, _sl=s_l), ["_se", "_sl"], [CUT])
    ep_t["score"] = np.where(ep_t["interval"] == 0, ep_t["_se"], ep_t["_sl"])
    ep_t["tier"] = np.asarray(score_tier(ep_t["score"].to_numpy()))
    ep_t["pt"] = ep_t["stop"] - ep_t["start"]
    tier_labels, rate_rows = {}, []
    for t in P.TIER_ORDER:
        m = ep_t["tier"] == t
        if not m.any():
            continue
        lo_s, hi_s = int(ep_t.loc[m, "score"].min()), int(ep_t.loc[m, "score"].max())
        rng_lbl = f"{lo_s}" if lo_s == hi_s else f"{lo_s}-{hi_s}"
        tier_labels[t] = f"{t} ({rng_lbl})"
        r, lo, hi = annual_rate(ep_t.loc[m, "pt"], ep_t.loc[m, "_ev"])
        rate_rows.append({"Tier": t, "Score_range": rng_lbl,
                          "N": int(ep_t.loc[m, "_id"].nunique()),
                          "Events": int(ep_t.loc[m, "_ev"].sum()),
                          "Person_years": float(ep_t.loc[m, "pt"].sum()),
                          "rate_pct": 100 * r, "lo_pct": 100 * lo, "hi_pct": 100 * hi})
    tv_rate_df = pd.DataFrame(rate_rows)
    tv_rate_df.round(3).to_csv("tv_risk_tiers.csv", index=False)
    print("\nTIME-VARYING SCORE TIERS (person-time allocated to current tier):")
    print(tv_rate_df.round(2).to_string(index=False))
    tn = ep_t.assign(tier_n=ep_t["tier"].map({"Low": 0, "Medium": 1, "High": 2}))
    trend = fit_ctv(tn, ["tier_n"]).summary.loc["tier_n"]
    print(f"HR per tier {trend['exp(coef)']:.2f}, p for trend {trend['p']:.3g}")
    lr_stat, p_logrank = logrank_counting(ep_t, "tier", list(P.TIER_ORDER))
    print(f"Log-rank test across time-updated tiers: chi2 {lr_stat:.1f} "
          f"({len(P.TIER_ORDER) - 1} df), p {p_logrank:.3g}")

    rng = np.random.default_rng(cfg.SEED)
    bs = []
    for _ in range(cfg.N_BOOT):
        idx = rng.choice(n, n, replace=True)
        if E[idx].sum() >= 5:
            bs.append(cindex_tv(T[idx], E[idx], s_e[idx], s_l[idx], CUT))
    stv_lo, stv_hi = np.percentile(bs, [2.5, 97.5])

    # Time-dependent AUC vs R2eD
    y_surv = Surv.from_arrays(event=E.astype(bool), time=T)
    auc_tv, _ = cumulative_dynamic_auc(y_surv, y_surv, td_tv, td_times)
    r2_feats = [f for f in cfg.R2ED_FEATS if f in df_k.columns]
    r2ed_risk, _, _ = nested_cv_risk(df_k[r2_feats].astype(float), surv_array(df_k), E,
                               cfg.OUTER_FOLDS, cfg.INNER_FOLDS, cfg.ALPHA_GRID, cfg.SEED)
    auc_r2, _ = cumulative_dynamic_auc(y_surv, y_surv, r2ed_risk, td_times)
    print(f"R2eD cross-validated C-index {cindex_tv(T, E, r2ed_risk, r2ed_risk, np.inf):.3f}")
    auc_tv_lo, auc_tv_hi = boot_auc(y_surv, E, td_tv, td_times, cfg.SEED)
    auc_r2_lo, auc_r2_hi = boot_auc(y_surv, E, r2ed_risk, td_times, cfg.SEED + 1)
    pd.DataFrame({"time_yr": td_times, "AUC_time_varying": auc_tv,
                  "AUC_tv_lo": auc_tv_lo, "AUC_tv_hi": auc_tv_hi,
                  "AUC_R2eD": auc_r2, "AUC_R2eD_lo": auc_r2_lo, "AUC_R2eD_hi": auc_r2_hi}).round(4).to_csv("tv_td_auc.csv", index=False)
    for name, a in [("time-varying", auc_tv), ("R2eD", auc_r2)]:
        print(f"td-AUC {name:<13} " + ", ".join(f"{t:g}yr {v:.3f}" for t, v in zip(td_times, a)))

    # Integer-score calibration slope
    print(f"Score points per fold (deep, prior early, prior late): {SP['pts']}")
    SCORE_SLOPE = cox_cal_slope(d, SP["lp_e"], SP["lp_l"], oof["tv_cut"])
    print(f"Integer-score calibration slope (Cox on OOF score LP): {SCORE_SLOPE[0]:.3f} "
          f"({SCORE_SLOPE[1]:.3f}-{SCORE_SLOPE[2]:.3f})")
    tier0 = np.asarray(score_tier(SP["se"]))

    # Score derivation variants
    o2, _, _ = km_observed(d["time"], d["event"], H)
    var_rows = []
    for var in cfg.SCORE_VARIANTS:
        V = sv[var]
        slp = cox_cal_slope(d, V["lp_e"], V["lp_l"], oof["tv_cut"])
        smv = smooth_calibration(d, V["risk"][H], H, n_boot=0)
        var_rows.append({
            "points_from": var[0], "recalibrated": var[1], "used_in_figures": var == primary,
            "C_index_cv": cindex_tv(T, E, V["se"], V["sl"], oof["tv_cut"]),
            "cal_slope": slp[0], "cal_slope_lo": slp[1], "cal_slope_hi": slp[2],
            f"E_O_{H:g}yr": float(np.nanmean(V["risk"][H])) / o2,
            f"ICI_{H:g}yr_pct": 100 * smv["ICI"], f"E90_{H:g}yr_pct": 100 * smv["E90"],
            "points_per_fold (deep, early, late)": str(V["pts"]),
            "shrink_per_fold": str(V["shrink"])})
    var_df = pd.DataFrame(var_rows)
    var_df.round(4).to_csv("tv_score_variants.csv", index=False)
    print("\nINTEGER SCORE DERIVATION VARIANTS (out-of-fold):")
    print(var_df.drop(columns=["points_per_fold (deep, early, late)", "shrink_per_fold"])
          .round(3).to_string(index=False))

    # Calibration summary at the main horizon
    smooth = {"model": {}, "score": {}}
    cal2 = calib_table(d, oof_h["tv"][H], H)
    sl2, ic2 = quartile_fit(cal2)
    smooth["model"][H] = smooth_calibration(d, oof_h["tv"][H], H)
    cal2.round(4).to_csv("tv_model_calibration_2yr.csv", index=False)
    pred2 = oof_h["tv"][H]
    calm = calibration_measures(d, pred2, H)
    sm2 = smooth["model"][H]

    fold_cal = []
    for k in sorted(np.unique(oof["fold"][np.isfinite(oof["fold"])])):
        m = oof["fold"] == k
        o_k, _, _ = km_observed(d.loc[m, "time"], d.loc[m, "event"], H)
        fold_cal.append({"fold": int(k), "events": int(E[m].sum()),
                         "C_index": float(folds.loc[folds["Fold"] == k, "C_time_varying"].iloc[0]),
                         "OE_2yr": o_k / float(np.nanmean(pred2[m]))})
    fold_cal = pd.DataFrame(fold_cal)
    fold_cal.round(4).to_csv("calibration_by_fold_2yr.csv", index=False)
    pd.DataFrame([{
        "horizon_yr": H, "n": n, "events_total": int(E.sum()),
        "observed_risk_KM": calm["observed"], "mean_predicted_risk": calm["expected"],
        "O_E": calm["OE"], "O_E_lo": calm["OE_lo"], "O_E_hi": calm["OE_hi"],
        "cal_slope_cloglog": calm["slope"], "cal_slope_cloglog_lo": calm["slope_lo"],
        "cal_slope_cloglog_hi": calm["slope_hi"],
        "cal_slope_LP": TV_SLOPE[0], "cal_slope_LP_lo": TV_SLOPE[1],
        "cal_slope_LP_hi": TV_SLOPE[2],
        "ICI": sm2["ICI"], "E50": sm2["E50"], "E90": sm2["E90"],
        "Brier_IPCW": calm["brier"], "C_index_pooled": C_TV,
        "C_index_fold_mean": fold_cal["C_index"].mean(),
        "C_index_fold_min": fold_cal["C_index"].min(),
        "C_index_fold_max": fold_cal["C_index"].max(),
        "O_E_fold_min": fold_cal["OE_2yr"].min(),
        "O_E_fold_max": fold_cal["OE_2yr"].max()}]
    ).round(4).to_csv("calibration_summary_2yr.csv", index=False)
    print(f"\nCALIBRATION SUMMARY at {H:g} yr (pooled out-of-fold, n={n}):")
    print(f"  O/E {calm['OE']:.2f} ({calm['OE_lo']:.2f}-{calm['OE_hi']:.2f}) | "
          f"slope {calm['slope']:.2f} ({calm['slope_lo']:.2f}-{calm['slope_hi']:.2f}) | "
          f"ICI {100 * sm2['ICI']:.2f}% | E90 {100 * sm2['E90']:.2f}% | Brier {calm['brier']:.4f}")

    # Main figure
    q1, q3 = np.percentile(T, [25, 75])
    p_txt = "< 0.001" if p_logrank < 0.001 else f"= {p_logrank:.3f}"
    km_info = (f"Log-rank p {p_txt}\n"
               f"Integer score C = {C_STV:.3f} (apparent; 95% CI {stv_lo:.3f}-{stv_hi:.3f})\n"
               f"Continuous model C = {C_TV:.3f} (cross-validated; 95% CI "
               f"{c_lo:.3f}-{c_hi:.3f})\n"
               f"N = {n}, events = {int(E.sum())}\n"
               f"Median FU = {np.median(T):.2f}y (IQR {q1:.2f}-{q3:.2f}; censored at treatment)")
    P.figure_longitudinal(
        draw_auc=lambda ax: P.panel_td_auc(ax, td_times, auc_tv, auc_tv_lo, auc_tv_hi,
                                           auc_r2, auc_r2_lo, auc_r2_hi, CUT),
        draw_cal=lambda ax: P.calibration_panel(
            ax, cal2, sl2, ic2, calm, sm2, H, cfg.FIG3_CAL_XMAX, cfg.FIG3_CAL_YMAX,
            cfg.CAL_TICK, T, E, pred2, cfg.FIG3_SHOW_BAND, cfg.FIG3_SHOW_HIST),
        draw_km=lambda ax: P.panel_km_tv(ax, ep_t, tier_labels, km_info, cfg.KM_T_MAX),
        rate_df=tv_rate_df, horizon=H)
    if cal2["obs_hi"].max() > cfg.FIG3_CAL_YMAX or cal2["pred"].max() > cfg.FIG3_CAL_XMAX:
        print("  NOTE: calibration panel axes clip a point or CI")
    print("Saved FIGURE_longitudinal_time_varying.png / .pdf")

    # Calibration at every horizon, model and score
    for kind in ("model", "score"):
        is_score = kind == "score"
        risk_by_h = oof_h["score"] if is_score else oof_h["tv"]
        stem = "tv_score_calibration" if is_score else "tv_calibration"
        hz_rows, cal_tabs = [], {}
        for h in cfg.CALIB_HORIZONS:
            cdf = (score_calib_table(d, risk_by_h[h], h, tier0, P.TIER_ORDER) if is_score
                   else calib_table(d, risk_by_h[h], h))
            cdf.insert(0, "horizon_yr", h)
            cal_tabs[h] = cdf
            sl, ic = quartile_fit(cdf)
            o_all, o_lo, o_hi = km_observed(d["time"], d["event"], h)
            e_all = float(np.nanmean(risk_by_h[h]))
            if h not in smooth[kind]:
                smooth[kind][h] = smooth_calibration(d, risk_by_h[h], h)
            sm = smooth[kind][h]
            hz_rows.append({"horizon_yr": h, "at_risk": int((T >= h).sum()),
                            "events_by_h": int(((T <= h) & (E == 1)).sum()),
                            "mean_pred": e_all, "KM_obs": o_all, "KM_obs_lo": o_lo,
                            "KM_obs_hi": o_hi,
                            "E_O_ratio": e_all / o_all if o_all > 0 else np.nan,
                            "group_slope": sl, "group_intercept": ic,
                            "ICI": sm["ICI"], "E50": sm["E50"], "E90": sm["E90"]})
        hz = pd.DataFrame(hz_rows)
        hz.round(4).to_csv(f"{stem}_horizons.csv", index=False)
        pd.concat(cal_tabs.values()).round(4).to_csv(f"{stem}_by_horizon.csv", index=False)
        print(f"\nCALIBRATION BY HORIZON ({kind}, out-of-fold):")
        print(hz.round(3).to_string(index=False))
        P.figure_calibration_horizons(
            hz, cal_tabs, smooth[kind], risk_by_h, SCORE_SLOPE if is_score else TV_SLOPE,
            T, E, stem, "integer score" if is_score else "time-varying model",
            f"FIGURE_{stem}_horizons", cfg.CAL_TICK, is_score,
            tiers=is_score and cfg.SCORE_CALIB_GROUPS == "tiers")

    # Supplement
    supp = os.path.join(out_dir, "supplement")
    os.makedirs(supp, exist_ok=True)
    header(f"SUPPLEMENT -> {supp}")
    sp = lambda name: os.path.join(supp, name)

    hs = [h for h in cfg.SUPP_HORIZONS if h in cfg.CALIB_HORIZONS]
    fig, axs = plt.subplots(1, len(hs), figsize=(7.2 * len(hs), 7.6), gridspec_kw={"wspace": 0.30})
    for i, (ax, h) in enumerate(zip(np.atleast_1d(axs), hs)):
        pred = oof_h["tv"][h]
        cdf = calib_table(d, pred, h)
        sl, ic = quartile_fit(cdf)
        main_h = h == H
        x_max = cfg.FIG3_CAL_XMAX if main_h else float(np.ceil(cdf["pred"].max() * 1.25 / cfg.CAL_TICK) * cfg.CAL_TICK)
        y_max = cfg.FIG3_CAL_YMAX if main_h else float(np.ceil(max(x_max, cdf["obs_hi"].max() * 1.05) / cfg.CAL_TICK) * cfg.CAL_TICK)
        P.calibration_panel(ax, cdf, sl, ic, calibration_measures(d, pred, h), smooth["model"][h],
                            h, x_max, y_max, cfg.CAL_TICK if y_max <= 0.3 else 0.05, T, E, pred,
                            cfg.FIG3_SHOW_BAND, cfg.FIG3_SHOW_HIST, legend_fs=8, at_risk_line=True)
        P.title(ax, chr(65 + i), f"Calibration at {h:g} years")
    P.save(fig, sp("FIGURE_S_calibration_2_5_10yr"))

    auc_at = dict(zip([float(t) for t in td_times], auc_tv))
    tab = []
    for h in cfg.CALIB_HORIZONS:
        cm = calibration_measures(d, oof_h["tv"][h], h)
        sm = smooth["model"][h]
        tab.append({
            "Horizon (yr)": f"{h:g}", "At risk": int((T >= h).sum()),
            "Ruptures by horizon": int(((T <= h) & (E == 1)).sum()),
            "Mean predicted risk (%)": round(100 * cm["expected"], 1),
            "Observed risk, KM (%)": round(100 * cm["observed"], 1),
            "O/E (95% CI)": f"{cm['OE']:.2f} ({cm['OE_lo']:.2f}-{cm['OE_hi']:.2f})",
            "Calibration slope (95% CI)": f"{cm['slope']:.2f} ({cm['slope_lo']:.2f}-{cm['slope_hi']:.2f})",
            "ICI (%)": round(100 * sm["ICI"], 2), "E50 (%)": round(100 * sm["E50"], 2),
            "E90 (%)": round(100 * sm["E90"], 2),
            "Time-dependent AUC": round(float(auc_at[h]), 3) if h in auc_at else np.nan})
    tab.append({"Horizon (yr)": "All follow-up", "At risk": n, "Ruptures by horizon": int(E.sum()),
                "Mean predicted risk (%)": np.nan, "Observed risk, KM (%)": np.nan,
                "O/E (95% CI)": "",
                "Calibration slope (95% CI)": f"{TV_SLOPE[0]:.2f} ({TV_SLOPE[1]:.2f}-{TV_SLOPE[2]:.2f})",
                "ICI (%)": np.nan, "E50 (%)": np.nan, "E90 (%)": np.nan,
                "Time-dependent AUC": np.nan})
    pd.DataFrame(tab).to_csv(sp("TABLE_S_calibration_by_horizon.csv"), index=False)

    risk_grp = pd.qcut(oof_h["tv"][H], 3, labels=["Low", "Medium", "High"]).astype(str)
    P.figure_observed_vs_predicted(T, E, np.asarray(risk_grp), td_tv, td_times,
                                   sp("FIGURE_S_observed_vs_predicted_over_time")
                                   ).round(4).to_csv(sp("TABLE_S_observed_vs_predicted_over_time.csv"),
                                                     index=False)
    for f_ in ("FIGURE_tv_calibration_horizons.png", "FIGURE_tv_calibration_horizons.pdf",
               "FIGURE_tv_score_calibration_horizons.png", "FIGURE_tv_score_calibration_horizons.pdf",
               "tv_calibration_horizons.csv", "tv_score_calibration_horizons.csv"):
        if os.path.exists(f_):
            shutil.copy(f_, sp(f_))
    print("Saved supplement figures and tables")

    # C-index summary: full cohort and conservatively managed subset
    le_full, ll_full = lp_pair(d, f_tv)
    r2_df = df_k[r2_feats + ["time", "event"]].astype(float)
    r2_df[r2_feats] = KNNImputer(n_neighbors=5).fit_transform(r2_df[r2_feats])
    r2_cph = CoxPHFitter(penalizer=0.01).fit(r2_df, "time", "event")
    r2ed_full = r2_cph.predict_log_partial_hazard(r2_df[r2_feats]).to_numpy()
    unt = df_k["untreated"].astype(bool).to_numpy() if "untreated" in df_k.columns else None
    oof_cut = oof["tv_cut"]
    entries = [
        ("Full cohort", "Time-varying model (continuous)", "cross-validated",
         (oof["tv_le"], oof["tv_ll"], oof_cut, None)),
        ("Full cohort", "Time-varying integer score", "cross-validated",
         (SP["se"], SP["sl"], oof_cut, None)),
        ("Full cohort", "Time-varying integer score", "apparent", (s_e, s_l, CUT, None)),
        ("Full cohort", "R2eD AVM score (refit)", "cross-validated",
         (r2ed_risk, r2ed_risk, np.inf, None))]
    has_unt = unt is not None and unt.sum() > 20 and E[unt].sum() >= 5
    if has_unt:
        cm_ = "Conservatively managed"
        entries += [
            (cm_, "Time-varying model (continuous)", "full-cohort model applied",
             (le_full, ll_full, CUT, unt)),
            (cm_, "Time-varying model (continuous)", "cross-validated",
             (oof["tv_le"], oof["tv_ll"], oof_cut, unt)),
            (cm_, "Time-varying integer score", "full-cohort points applied", (s_e, s_l, CUT, unt)),
            (cm_, "Time-varying integer score", "cross-validated", (SP["se"], SP["sl"], oof_cut, unt)),
            (cm_, "R2eD AVM score (refit)", "full-cohort model applied",
             (r2ed_full, r2ed_full, np.inf, unt)),
            (cm_, "R2eD AVM score (refit)", "cross-validated", (r2ed_risk, r2ed_risk, np.inf, unt))]
    rows = []
    for cohort, model, kind, (a, b, c, mask) in entries:
        cc, lo, hi, nn, ev = boot_c(T, E, a, b, c, mask)
        rows.append({"Cohort": cohort, "Model": model, "Estimate": kind, "N": nn,
                     "Events": ev, "C_index": cc, "CI_low": lo, "CI_high": hi})
    csum = pd.DataFrame(rows)
    csum.round(4).to_csv("tv_cindex_summary.csv", index=False)
    print("\nC-INDEX SUMMARY (bootstrap 95% CI):")
    print(csum.round(3).to_string(index=False))

    # td-AUC in the conservatively managed subset
    if has_unt:
        d_u = d.loc[unt].reset_index(drop=True)
        y_u = Surv.from_arrays(event=E[unt].astype(bool), time=T[unt])
        t_u = np.array([t for t in td_times if T[unt][E[unt] == 1].min() < t < T[unt].max()])
        unt_rows = []
        for kind, marker in [
                ("full-cohort model applied",
                 np.column_stack([predict_risk(d_u, f_tv, t) for t in t_u])),
                ("cross-validated", td_tv[unt][:, [list(td_times).index(t) for t in t_u]])]:
            for k, t in enumerate(t_u):
                try:
                    a = float(cumulative_dynamic_auc(y_surv, y_u, marker[:, k], [t])[0][0])
                except Exception:
                    a = np.nan
                unt_rows.append({"Estimate": kind, "time_yr": float(t), "AUC": a})
        unt_auc = pd.DataFrame(unt_rows).pivot(index="time_yr", columns="Estimate",
                                               values="AUC").reset_index()
        unt_auc.round(4).to_csv("tv_td_auc_conservatively_managed.csv", index=False)
        print("\nTIME-DEPENDENT AUC, CONSERVATIVELY MANAGED:")
        print(unt_auc.round(3).to_string(index=False))

    # Fairness: C-index by race
    r = df_k["Race_nonwhite"].to_numpy(float) if "Race_nonwhite" in df_k.columns else np.full(n, np.nan)
    subgroups = [("White", r == 0), ("Non-white", r == 1)]
    fair = []
    for name, mask in subgroups:
        mask = np.asarray(mask, bool)
        ev = int(E[mask].sum())
        if mask.sum() < 20 or ev < 5:
            fair.append({"Subgroup": name, "N": int(mask.sum()), "Events": ev, "C_index": np.nan,
                         "CI_low": np.nan, "CI_high": np.nan, "Note": "too few events"})
            continue
        cc, lo, hi, nn, ev = boot_c(T, E, oof["tv_le"], oof["tv_ll"], oof_cut, mask)
        fair.append({"Subgroup": name, "N": nn, "Events": ev, "C_index": cc,
                     "CI_low": lo, "CI_high": hi,
                     "Note": "few events, interpret cautiously" if ev < 30 else ""})
    fair = pd.DataFrame(fair)
    fair.round(4).to_csv("tv_fairness_subgroups.csv", index=False)
    print("\nFAIRNESS: CROSS-VALIDATED C-INDEX, WHITE VS NON-WHITE:")
    print(fair.round(3).to_string(index=False))

    # Sensitivity: wider penalty grid
    header(f"SENSITIVITY: WIDER PENALTY GRID {cfg.WIDE_PENALIZERS}")
    w_le, w_ll, w_cut = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    w_pens = []
    for f, (tr, te) in enumerate(outer.split(d, d["event"])):
        train = d.iloc[tr].reset_index(drop=True)
        test = d.iloc[te].reset_index(drop=True)
        cut = select_cut(train)
        p_w = tune_pen(train, cut, cfg.SEED + f, grid=cfg.WIDE_PENALIZERS)
        w_le[te], w_ll[te] = lp_pair(test, fit_model(train, cut, p_w))
        w_cut[te] = cut
        w_pens.append(p_w)
    cw, cw_lo, cw_hi, _, _ = boot_c(T, E, w_le, w_ll, w_cut)
    slope_w = cox_cal_slope(d, w_le, w_ll, w_cut)
    prim_pens = folds["pen_tv"].tolist()
    pd.DataFrame([
        {"Grid": f"primary {cfg.PENALIZERS}", "penalizer_per_fold": str(prim_pens),
         "C_index_cv": C_TV, "C_lo": c_lo, "C_hi": c_hi,
         "cal_slope": TV_SLOPE[0], "cal_slope_lo": TV_SLOPE[1], "cal_slope_hi": TV_SLOPE[2]},
        {"Grid": f"wide {cfg.WIDE_PENALIZERS}", "penalizer_per_fold": str(w_pens),
         "C_index_cv": cw, "C_lo": cw_lo, "C_hi": cw_hi,
         "cal_slope": slope_w[0], "cal_slope_lo": slope_w[1], "cal_slope_hi": slope_w[2]}]
    ).round(4).to_csv("tv_penalty_grid_sensitivity.csv", index=False)
    print(f"  selected per fold: primary {prim_pens} | wide {w_pens}")
    print(f"  C-index: primary {C_TV:.3f} | wide {cw:.3f} [{cw_lo:.3f}, {cw_hi:.3f}]")
    edge = [p for p in w_pens if p in (min(cfg.WIDE_PENALIZERS), max(cfg.WIDE_PENALIZERS))]
    if edge:
        print(f"  NOTE: {len(edge)} fold(s) selected a value at the edge of the wide grid {edge}")

    # Summary
    e = coef_df[coef_df["Term"].str.contains("0-")].iloc[0]
    l = coef_df[coef_df["Term"].str.contains(">")].iloc[0]
    header("SUMMARY")
    print(f"Prior rupture HR {e['HR']:.2f} ({e['HR_lo']:.2f}-{e['HR_hi']:.2f}) in the first "
          f"{CUT:g} yr vs {l['HR']:.2f} ({l['HR_lo']:.2f}-{l['HR_hi']:.2f}) thereafter "
          f"(Wald p = {P_WALD:.3f}).")
    print(f"Cross-validated C-index {C_TV:.3f} (95% CI {c_lo:.3f}-{c_hi:.3f}).")
    print(f"Cut-points selected across folds: {folds['cut_yr'].tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=cfg.DATA_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"Registry not found: {args.data} (set DATA_PATH in config.py)")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.data)),
                                   f"results_{datetime.now():%Y%m%d_%H%M}")
    main(args.data, os.path.abspath(out))