"""Calibration measures: grouped KM tables, Cox slopes, O/E, Brier and smoothed curves (ICI/E50/E90)."""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import linregress
from sksurv.metrics import brier_score
from sksurv.util import Surv

import config as cfg
from .episodes import fit_ctv, split
from .stats import km_observed


def _TE(d):
    return d["time"].to_numpy(float), d["event"].to_numpy().astype(int)


def calib_table(d, risk, h):
    """KM observed vs mean predicted risk in quantiles of predicted risk."""
    q = pd.qcut(risk, cfg.CALIB_QUANTILES, labels=False, duplicates="drop")
    rows = []
    for k in np.unique(q):
        m = q == k
        obs, lo, hi = km_observed(d.loc[m, "time"], d.loc[m, "event"], h)
        rows.append({"quartile": int(k) + 1, "n": int(m.sum()),
                     "pred": float(risk[m].mean()), "obs": obs,
                     "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)


def score_calib_table(d, risk, h, tier0, tier_order):
    """Same as calib_table for the integer score, by rank quartile or by tier."""
    if cfg.SCORE_CALIB_GROUPS == "quartiles":
        q = pd.qcut(pd.Series(risk).rank(method="first"), cfg.CALIB_QUANTILES,
                    labels=False).to_numpy()
        rows = []
        for k in range(cfg.CALIB_QUANTILES):
            m = q == k
            obs, lo, hi = km_observed(d.loc[m, "time"], d.loc[m, "event"], h)
            rows.append({"quartile": k + 1, "n": int(m.sum()),
                         "pred": float(risk[m].mean()), "obs": obs,
                         "obs_lo": lo, "obs_hi": hi})
        return pd.DataFrame(rows)
    rows = []
    for k, t in enumerate(tier_order):
        m = tier0 == t
        if m.sum() < 5:
            continue
        obs, lo, hi = km_observed(d.loc[m, "time"], d.loc[m, "event"], h)
        rows.append({"tier": t, "quartile": k + 1, "n": int(m.sum()),
                     "pred": float(risk[m].mean()), "obs": obs,
                     "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)


def quartile_fit(cdf):
    if len(cdf) < 2:
        return np.nan, np.nan
    r = linregress(cdf["pred"], cdf["obs"])
    return float(r.slope), float(r.intercept)


def cox_cal_slope(d, le, ll, cut_arr):
    """Calibration slope: Cox model on the out-of-fold time-varying LP."""
    cut_arr = np.asarray(cut_arr, float)
    cuts = np.unique(cut_arr[np.isfinite(cut_arr)])
    dd = d.assign(_le=le, _ll=ll, _cut=cut_arr)
    ep = split(dd, ["_le", "_ll", "_cut"], cuts)
    ep["lp"] = np.where(ep["stop"] <= ep["_cut"] + 1e-9, ep["_le"], ep["_ll"])
    s = fit_ctv(ep, ["lp"]).summary.loc["lp"]
    return float(s["coef"]), float(s["coef lower 95%"]), float(s["coef upper 95%"])


def _rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell)."""
    x, k = np.asarray(x, float), np.asarray(knots, float)
    cols = [x]
    if len(k) >= 3:
        norm = (k[-1] - k[0]) ** 2
        for j in range(len(k) - 2):
            cols.append((np.clip(x - k[j], 0, None) ** 3
                         - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                         + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
                        / norm)
    return np.column_stack(cols)


def smooth_calibration(d, pred, h, seed=None, n_boot=None):
    """Smoothed calibration curve (Austin 2020) with bootstrap band and ICI/E50/E90."""
    seed = cfg.SEED if seed is None else seed
    n_boot = cfg.N_BOOT_CAL if n_boot is None else n_boot
    T, E = _TE(d)
    n = len(T)
    p = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
    cll = np.log(-np.log(1 - p))
    Th = np.minimum(T, h)
    Eh = ((E == 1) & (T <= h)).astype(int)
    qs = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[cfg.CAL_KNOTS]
    knots = np.unique(np.percentile(cll, qs))
    grid_p = np.linspace(np.percentile(p, 1), np.percentile(p, 99), 100)
    grid_c = np.log(-np.log(1 - grid_p))

    def fit_pred(idx, evals):
        B = _rcs_basis(cll[idx], knots)
        cols = [f"s{j}" for j in range(B.shape[1])]
        dfc = pd.DataFrame(B, columns=cols)
        dfc["T"], dfc["E"] = Th[idx], Eh[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=1e-4).fit(dfc, "T", "E")
        out = []
        for xc in evals:
            Bx = pd.DataFrame(_rcs_basis(xc, knots), columns=cols)
            out.append(1 - cph.predict_survival_function(Bx, times=[h]).to_numpy().ravel())
        return out

    obs_i, obs_g = fit_pred(np.arange(n), [cll, grid_c])
    diff = np.abs(obs_i - p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if Eh[idx].sum() < 5:
            continue
        try:
            boots.append(fit_pred(idx, [grid_c])[0])
        except Exception:
            continue
    boots = np.vstack(boots) if boots else None
    return {"grid": grid_p, "obs": obs_g,
            "lo": None if boots is None else np.nanpercentile(boots, 2.5, axis=0),
            "hi": None if boots is None else np.nanpercentile(boots, 97.5, axis=0),
            "ICI": float(diff.mean()), "E50": float(np.median(diff)),
            "E90": float(np.percentile(diff, 90))}


def calibration_measures(d, pred, h):
    """Cloglog calibration slope, O/E and IPCW Brier at horizon h."""
    T, E = _TE(d)
    p = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
    cll = np.log(-np.log(1 - p))
    Th = np.minimum(T, h)
    Eh = ((E == 1) & (T <= h)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = CoxPHFitter().fit(pd.DataFrame({"x": cll, "T": Th, "E": Eh}),
                              "T", "E").summary.loc["x"]
    obs, o_lo, o_hi = km_observed(d["time"], d["event"], h)
    exp_ = float(p.mean())
    try:
        y_s = Surv.from_arrays(E.astype(bool), T)
        _, bs = brier_score(y_s, y_s, (1 - p).reshape(-1, 1), [h])
        brier = float(bs[0])
    except Exception:
        brier = np.nan
    return {"slope": float(s["coef"]), "slope_lo": float(s["coef lower 95%"]),
            "slope_hi": float(s["coef upper 95%"]),
            "OE": obs / exp_, "OE_lo": o_lo / exp_, "OE_hi": o_hi / exp_,
            "observed": obs, "expected": exp_, "brier": brier}
