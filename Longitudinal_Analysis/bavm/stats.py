"""Shared statistics: ridge Cox pipeline, nested CV, event rates, KM risk and the integer score."""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import chi2
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


def build_pipe(alpha=1.0):
    """KNN imputation, scaling and ridge Cox, all refit inside each training fold."""
    return Pipeline([
        ("imp", KNNImputer(n_neighbors=5)),
        ("scl", StandardScaler()),
        ("model", CoxPHSurvivalAnalysis(alpha=alpha)),
    ])


def cindex(event, time, risk):
    try:
        return concordance_index_censored(
            np.asarray(event).astype(bool),
            np.asarray(time).astype(float),
            np.asarray(risk).astype(float))[0]
    except Exception:
        return np.nan


def nested_cv_risk(X, y, event_label, n_outer, n_inner, alphas, seed):
    """Out-of-fold ridge Cox risk, per-fold C-index and chosen alphas; inner folds pick alpha."""
    n_outer = min(n_outer, max(2, int(event_label.sum()) // 3))
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan)
    fold_c, chosen = [], []
    for otr, ots in outer.split(X, event_label):
        ev_tr = event_label[otr]
        n_in = min(n_inner, max(2, int(ev_tr.sum()) // 3))
        inner = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=seed)
        best_a, best_s = alphas[0], -np.inf
        for a in alphas:
            cs = []
            for ti, vi in inner.split(X.iloc[otr], ev_tr):
                try:
                    p = build_pipe(a).fit(X.iloc[otr].iloc[ti], y[otr][ti])
                    yv = y[otr][vi]
                    cs.append(cindex(yv["event"], yv["time"],
                                     p.predict(X.iloc[otr].iloc[vi])))
                except Exception:
                    continue
            cs = [c for c in cs if not np.isnan(c)]
            if cs and np.mean(cs) > best_s:
                best_s, best_a = np.mean(cs), a
        pipe = build_pipe(best_a).fit(X.iloc[otr], y[otr])
        oof[ots] = pipe.predict(X.iloc[ots])
        yo = y[ots]
        fold_c.append(cindex(yo["event"], yo["time"], oof[ots]))
        chosen.append(best_a)
    return oof, np.array(fold_c, dtype=float), chosen


def km_observed(time, event, t_eval):
    """KM cumulative incidence at t_eval with 95% limits."""
    kmf = KaplanMeierFitter().fit(time, event)
    sf = float(kmf.survival_function_at_times(t_eval).values[0])
    ci = kmf.confidence_interval_survival_function_
    pos = int(np.abs(np.asarray(ci.index, dtype=float) - t_eval).argmin())
    row = ci.iloc[pos]
    return 1 - sf, max(1 - row.iloc[1], 0.0), min(1 - row.iloc[0], 1.0)


def annual_rate(time, event):
    """Events per person-year with a Poisson 95% interval."""
    py = float(np.sum(time))
    ev = int(np.sum(event))
    if py <= 0:
        return np.nan, np.nan, np.nan
    if ev == 0:
        return 0.0, 0.0, 3.0 / py
    lo = 0.5 * chi2.ppf(0.025, 2 * ev) / py
    hi = 0.5 * chi2.ppf(0.975, 2 * ev + 2) / py
    return ev / py, lo, hi


def integer_score(df, age_cuts=(20, 40, 60), age_pts=(0, 1, 2, 2),
                  pts_deep=2, pts_prior=1):
    """Age bracket points + exclusive deep location + prior rupture."""
    age = pd.to_numeric(df["Age_dx"], errors="coerce")
    bracket = np.digitize(age, age_cuts)
    p_age = np.asarray(age_pts, dtype=float)[bracket]
    p_age[age.isna()] = np.nan
    return (p_age
            + pts_deep * pd.to_numeric(df["Deep_Loc_excl"], errors="coerce").fillna(0)
            + pts_prior * pd.to_numeric(df["Ever_Ruptured"], errors="coerce").fillna(0))


def score_tier(score, cuts=(1, 3)):
    """Low 0-1, Medium 2-3, High 4+."""
    lo, mid = cuts
    return np.select([score <= lo, score <= mid], ["Low", "Medium"],
                     default="High")