"""
Statistical helpers for the longitudinal analysis
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

RANDOM_STATE = 42
ALPHA_GRID = [0.1, 0.5, 1.0, 5.0]
def build_pipe(alpha=1.0):
    """Ridge Cox with imputation and scaling inside the pipeline
    """
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

def bootstrap_cindex(event, time, risk, n_boot=1000, seed=RANDOM_STATE):
    """Percentile CI for the C-index by resampling patients"""
    rng = np.random.default_rng(seed)
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    risk = np.asarray(risk).astype(float)
    n = len(event)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if event[idx].sum() < 2:
            continue
        c = cindex(event[idx], time[idx], risk[idx])
        if not np.isnan(c):
            vals.append(c)
    if not vals:
        return np.nan, np.nan, np.nan
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))

def nested_cv_risk(X, y, event_label, n_outer=5, n_inner=3,
                   alphas=ALPHA_GRID, seed=RANDOM_STATE, horizon=None,
                   fixed_alpha=None):
    """Nested CV over alpha. Returns out-of-fold risk, per-fold C, chosen alphas
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    n_outer = min(n_outer, max(2, int(event_label.sum()) // 3))
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof_risk = np.full(len(X), np.nan)
    oof_prob = np.full(len(X), np.nan)
    fold_c, chosen = [], []
    failures = 0

    for otr, ots in outer.split(X, event_label):
        ev_tr = event_label[otr]
        n_in = min(n_inner, max(2, int(ev_tr.sum()) // 3))
        inner = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=seed)

        best_a, best_s = alphas[0], -np.inf
        for a in alphas:
            cs = []
            for ti, vi in inner.split(X.iloc[otr], ev_tr):
                p = clone(build_pipe(a))
                try:
                    p.fit(X.iloc[otr].iloc[ti], y[otr][ti])
                    cs.append(cindex([s[0] for s in y[otr][vi]],
                                     [s[1] for s in y[otr][vi]],
                                     p.predict(X.iloc[otr].iloc[vi])))
                except Exception:
                    failures += 1
            cs = [c for c in cs if not np.isnan(c)]
            if cs and np.mean(cs) > best_s:
                best_s, best_a = np.mean(cs), a

        pipe = build_pipe(best_a)
        pipe.fit(X.iloc[otr], y[otr])
        r = pipe.predict(X.iloc[ots])
        oof_risk[ots] = r

        if horizon is not None:
            if fixed_alpha is not None and fixed_alpha != best_a:
                pipe = build_pipe(fixed_alpha).fit(X.iloc[otr], y[otr])
            try:
                Xt = X.iloc[ots]
                for name, step in pipe.steps[:-1]:
                    Xt = step.transform(Xt)
                fns = pipe.named_steps["model"].predict_survival_function(Xt)
                oof_prob[ots] = [1.0 - float(f(min(horizon, f.x.max())))
                                 for f in fns]
            except Exception as e:
                print(f"  fold survival function failed: {e}")
        fold_c.append(cindex([s[0] for s in y[ots]],
                             [s[1] for s in y[ots]], r))
        chosen.append(best_a)

    if failures:
        print(f"  note: {failures} inner fits failed to converge and were dropped")
    if horizon is not None:
        return oof_risk, np.array(fold_c, dtype=float), chosen, oof_prob
    return oof_risk, np.array(fold_c, dtype=float), chosen


def td_auc(y_train, y_test, risk, times):
    """Cumulative time-dependent AUC over a grid of horizons"""
    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        return np.asarray(auc), float(mean_auc)
    except Exception as e:
        print(f"  td-AUC failed: {e}")
        return np.full(len(times), np.nan), np.nan


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
    rate = ev / py
    if ev == 0:
        return 0.0, 0.0, 3.0 / py
    lo = 0.5 * _chi2_ppf(0.025, 2 * ev) / py
    hi = 0.5 * _chi2_ppf(0.975, 2 * ev + 2) / py
    return rate, lo, hi

def _chi2_ppf(q, dfree):
    from scipy import stats as sp_stats
    return sp_stats.chi2.ppf(q, dfree)

def derive_integer_score(betas, feature_names, age_mean, age_sd,
                         age_range=(0, 100), out_csv="score_derivation.csv"):
    """Turn the ridge coefficients into an integer point system
    """
    b = dict(zip(feature_names, np.asarray(betas, dtype=float)))
    nonzero = {k: v for k, v in b.items() if abs(v) > 1e-8}
    if not nonzero:
        raise ValueError("All coefficients are zero - nothing to scale.")
    unit = min(abs(v) for v in nonzero.values())

    rows, cuts = [], []
    for f, coef in b.items():
        if f == "Age_dx":
            step_sd = unit / abs(coef) if abs(coef) > 1e-8 else np.inf
            step_yr = step_sd * age_sd
            n_steps = 3
            cuts = [age_mean + (k - (n_steps - 1) / 2) * step_yr
                    for k in range(n_steps)]
            cuts = [float(np.clip(c, *age_range)) for c in cuts]
            rows.append({"Feature": f, "beta": round(coef, 4),
                         "points": f"0-{n_steps}",
                         "detail": "cut at " + ", ".join(f"{c:.0f}" for c in cuts)})
        else:
            rows.append({"Feature": f, "beta": round(coef, 4),
                         "points": int(round(coef / unit)),
                         "detail": "present = points, absent = 0"})

    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"Saved {out_csv}")
    return cuts, table

def integer_score(df, age_cuts=(27, 45, 63)):
    """Age bracket + deep location + prior rupture.
    """
    age = pd.to_numeric(df["Age_dx"], errors="coerce")
    c = sorted(age_cuts)
    p_age = np.select([age < c[0], age < c[1], age < c[2]],
                      [0, 1, 2], default=3).astype(float)
    p_age[age.isna()] = np.nan
    return (p_age
            + pd.to_numeric(df["Deep_Loc_excl"], errors="coerce").fillna(0)
            + pd.to_numeric(df["Ever_Ruptured"], errors="coerce").fillna(0))


def score_tier(score, cuts=(1, 3)):
    """Collapse the integer score into tiers.
    Default Low 0-1, Medium 2-3, High 4+.
    """
    lo, mid = cuts
    return np.select([score <= lo, score <= mid], ["Low", "Medium"],
                     default="High")


def followup_summary(df):
    """Follow-up duration and treatment split, for the cohort description."""
    t = pd.to_numeric(df["time"], errors="coerce")
    unt = df["untreated"].astype(bool)
    return {
        "N": len(df),
        "Events": int(df["event"].sum()),
        "Event_rate_pct": round(100 * df["event"].mean(), 1),
        "Followup_mean_yr": round(float(t.mean()), 2),
        "Followup_median_yr": round(float(t.median()), 2),
        "Followup_IQR": f"{t.quantile(0.25):.2f}-{t.quantile(0.75):.2f}",
        "Person_years_total": round(float(t.sum()), 1),
        "N_treated": int((~unt).sum()),
        "Pct_treated": round(100 * (~unt).mean(), 1),
        "Events_treated": int(df.loc[~unt, "event"].sum()),
        "Followup_mean_treated_yr": round(float(t[~unt].mean()), 2),
        "N_untreated": int(unt.sum()),
        "Pct_untreated": round(100 * unt.mean(), 1),
        "Events_untreated": int(df.loc[unt, "event"].sum()),
        "Followup_mean_untreated_yr": round(float(t[unt].mean()), 2),
        "Followup_median_untreated_yr": round(float(t[unt].median()), 2),
        "Followup_IQR_untreated": (f"{t[unt].quantile(0.25):.2f}-"
                                   f"{t[unt].quantile(0.75):.2f}"),
    }