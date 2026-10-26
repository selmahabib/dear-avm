"""Time-varying ridge Cox model (prior rupture split at a cut-point) and its integer score."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config as cfg
from .episodes import fit_ctv, split
from .explore import cutpoint_scan
from .stats import integer_score


def _terms(ep, cut, sd):
    out = pd.DataFrame({"_id": ep["_id"], "start": ep["start"],
                        "stop": ep["stop"], "_ev": ep["_ev"]})
    out["age"] = ep[cfg.AGE] / sd[cfg.AGE]
    out["deep"] = ep[cfg.DEEP] / sd[cfg.DEEP]
    if cut is None:
        out["prior"] = ep[cfg.PRIOR] / sd[cfg.PRIOR]
        return out, ["age", "deep", "prior"]
    out["prior_early"] = ep[cfg.PRIOR] / sd[cfg.PRIOR] * (ep["interval"] == 0)
    out["prior_late"] = ep[cfg.PRIOR] / sd[cfg.PRIOR] * (ep["interval"] == 1)
    return out, ["age", "deep", "prior_early", "prior_late"]


def fit_model(train, cut, pen, scale=True):
    """Fit the model; cut=None gives the constant-HR model. Returns raw-scale coefficients."""
    sd = (train[cfg.BASE].std(ddof=0).replace(0, 1) if scale
          else pd.Series(1.0, index=cfg.BASE))
    ep = split(train, cfg.BASE, [] if cut is None else [cut])
    X, cols = _terms(ep, cut, sd)
    ctv = fit_ctv(X, cols, penalizer=pen)
    b = ctv.params_[cols]
    m = {"b_age": b["age"] / sd[cfg.AGE], "b_deep": b["deep"] / sd[cfg.DEEP],
         "pen": pen, "ctv": ctv, "cols": cols}
    if cut is None:
        m.update(b_early=b["prior"] / sd[cfg.PRIOR], b_late=b["prior"] / sd[cfg.PRIOR],
                 cut=np.inf)
    else:
        m.update(b_early=b["prior_early"] / sd[cfg.PRIOR],
                 b_late=b["prior_late"] / sd[cfg.PRIOR], cut=float(cut))
    le, ll = lp_pair(train, m)
    m["tk"], m["dH"] = breslow(train["time"], train["event"], le, ll, m["cut"])
    return m


def lp_pair(dd, m):
    """Linear predictor before and after the cut-point."""
    base = m["b_age"] * dd[cfg.AGE].to_numpy() + m["b_deep"] * dd[cfg.DEEP].to_numpy()
    pr = dd[cfg.PRIOR].to_numpy()
    return base + m["b_early"] * pr, base + m["b_late"] * pr


def breslow(T, E, le, ll, cut):
    """Baseline hazard increments at each event time."""
    T, E = np.asarray(T, float), np.asarray(E, int)
    tk = np.unique(T[E == 1])
    dH = np.empty(len(tk))
    for k, t in enumerate(tk):
        at = T >= t
        lp = le[at] if t <= cut else ll[at]
        dH[k] = ((T == t) & (E == 1)).sum() / np.exp(lp).sum()
    return tk, dH


def predict_generic(le, ll, tk, dH, cut, horizon):
    H_e = dH[tk <= min(horizon, cut)].sum()
    H_l = dH[(tk > cut) & (tk <= horizon)].sum()
    return 1 - np.exp(-(np.exp(le) * H_e + np.exp(ll) * H_l))


def predict_risk(dd, m, horizon):
    """Predicted rupture risk by `horizon` years."""
    le, ll = lp_pair(dd, m)
    return predict_generic(le, ll, m["tk"], m["dH"], m["cut"], horizon)


def cindex_tv(T, E, le, ll, cut):
    """Harrell's C with each patient's LP evaluated at each event time."""
    T, E = np.asarray(T, float), np.asarray(E, int)
    cut = np.broadcast_to(np.asarray(cut, float), T.shape)
    num = den = 0.0
    for i in np.flatnonzero(E == 1):
        t = T[i]
        j = T > t
        nj = j.sum()
        if nj == 0:
            continue
        lp_t = np.where(t <= cut, le, ll)
        diff = lp_t[i] - lp_t[j]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += nj
    return num / den if den else np.nan


def partial_loglik(T, E, le, ll, cut):
    """Held-out Cox partial log-likelihood (Breslow ties)."""
    T, E = np.asarray(T, float), np.asarray(E, int)
    out = 0.0
    for t in np.unique(T[E == 1]):
        at = T >= t
        lp = le if t <= cut else ll
        dk = (T == t) & (E == 1)
        out += lp[dk].sum() - dk.sum() * np.log(np.exp(lp[at]).sum())
    return out


def select_cut(train):
    if cfg.FIXED_CUT is not None:
        return float(cfg.FIXED_CUT)
    scan = cutpoint_scan(train, cfg.PRIOR, [cfg.AGE, cfg.DEEP], cfg.CUT_GRID,
                         "time", "event")
    if scan.empty:
        return 1.0
    return float(scan.loc[scan["LR_chi2"].idxmax(), "cut_yr"])


def tune_pen(train, cut, seed, grid=None):
    """Pick the ridge penalizer by inner CV."""
    grid = cfg.PENALIZERS if grid is None else grid
    skf = StratifiedKFold(cfg.INNER_FOLDS, shuffle=True, random_state=seed)
    scores = {p: [] for p in grid}
    for tr, va in skf.split(train, train["event"]):
        a = train.iloc[tr].reset_index(drop=True)
        b = train.iloc[va].reset_index(drop=True)
        for p in grid:
            try:
                m = fit_model(a, cut, p)
                le, ll = lp_pair(b, m)
                if cfg.TUNE_BY == "loglik":
                    scores[p].append(partial_loglik(b["time"], b["event"], le, ll, m["cut"]))
                else:
                    scores[p].append(cindex_tv(b["time"], b["event"], le, ll, m["cut"]))
            except Exception:
                scores[p].append(np.nan)
    mean = {p: np.nanmean(v) if np.isfinite(v).any() else -np.inf
            for p, v in scores.items()}
    return max(grid, key=lambda p: mean[p])


# Integer score

def points_from(m):
    """Sullivan points (deep, prior early, prior late); one point = one age bracket."""
    B = cfg.AGE_STEP_YR * abs(m["b_age"])
    return (max(0, int(round(m["b_deep"] / B))),
            max(0, int(round(m["b_early"] / B))),
            max(0, int(round(m["b_late"] / B))))


def derive_points(train, cut, pen=0.0):
    return points_from(fit_model(train, cut, pen, scale=pen > 0))


def score_with(dd, pts_deep, pts_prior):
    return np.asarray(integer_score(dd[cfg.BASE], age_cuts=cfg.MARS_CUTS,
                                    age_pts=cfg.AGE_PTS, pts_deep=pts_deep,
                                    pts_prior=pts_prior), float)


def tv_scores(dd, pts):
    """Integer score before and after the cut-point."""
    pdp, pe, pl = pts
    return score_with(dd, pdp, pe), score_with(dd, pdp, pl)


def fit_score_model(train, se, sl, cut, shrink=1.0):
    """Cox model on the time-varying score, coefficient scaled by `shrink`."""
    ep = split(train.assign(_se=se, _sl=sl), ["_se", "_sl"], [cut])
    ep["sc"] = np.where(ep["interval"] == 0, ep["_se"], ep["_sl"])
    b = float(fit_ctv(ep, ["sc"]).params_["sc"]) * shrink
    tk, dH = breslow(train["time"], train["event"], b * se, b * sl, cut)
    return {"b": b, "tk": tk, "dH": dH, "cut": cut, "shrink": shrink}


def inner_shrink(train, cut, pen, seed):
    """Inner-CV calibration slope of the score model, capped to [0.3, 1]."""
    skf = StratifiedKFold(cfg.INNER_FOLDS, shuffle=True, random_state=seed)
    slopes = []
    for tr, va in skf.split(train, train["event"]):
        a = train.iloc[tr].reset_index(drop=True)
        b_ = train.iloc[va].reset_index(drop=True)
        try:
            pts = derive_points(a, cut, pen)
            se_a, sl_a = tv_scores(a, pts)
            se_b, sl_b = tv_scores(b_, pts)
            m = fit_score_model(a, se_a, sl_a, cut)
            ep = split(b_.assign(_le=m["b"] * se_b, _ll=m["b"] * sl_b),
                       ["_le", "_ll"], [cut])
            ep["lp"] = np.where(ep["interval"] == 0, ep["_le"], ep["_ll"])
            slopes.append(float(fit_ctv(ep, ["lp"]).params_["lp"]))
        except Exception:
            continue
    return float(np.clip(np.mean(slopes), 0.3, 1.0)) if slopes else 1.0
