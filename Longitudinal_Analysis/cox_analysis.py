"""
Univariate and multivariable Cox regression 
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.outliers_influence import variance_inflation_factor

def _bh(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = len(pm)
    if n == 0:
        return p.copy()
    order = np.argsort(pm)
    ranked = pm[order]
    q = np.empty_like(pm)
    cmin = 1.0
    for i in range(n - 1, -1, -1):
        cmin = min(cmin, ranked[i] * n / (i + 1))
        q[order[i]] = cmin
    out = p.copy()
    out[mask] = np.clip(q, 0, 1)
    return out

def univariate_cox(df, features, out_csv="cox_univariate.csv"):
    """Candidate Feature screening
    """
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        sub = df[[f, "time", "event"]].dropna()
        if sub[f].nunique() < 2 or sub["event"].sum() < 3:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="time", event_col="event")
            s = cph.summary.loc[f]
            rows.append({
                "Feature": f,
                "N_at_risk": len(sub),
                "Events": int(sub["event"].sum()),
                "HR": round(float(s["exp(coef)"]), 3),
                "CI_low": round(float(s["exp(coef) lower 95%"]), 3),
                "CI_high": round(float(s["exp(coef) upper 95%"]), 3),
                "p": float(s["p"]),
            })
        except Exception as e:
            print(f"  {f}: did not converge ({type(e).__name__})")
    out = pd.DataFrame(rows)
    if len(out):
        out["q_BH"] = _bh(out["p"].to_numpy())
        out = out.sort_values("p").reset_index(drop=True)
        out.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
    return out


def multivariable_cox(df, features, out_csv="cox_multivariable.csv"):
    """Multivariable Cox on the screened features with VIF and Schoenfeld
    Returns (summary_table, vif_table, schoenfeld_table).
    """
    sub = df[list(features) + ["time", "event"]].dropna()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="time", event_col="event")
    s = cph.summary
    out = pd.DataFrame({
        "Feature": s.index,
        "HR": s["exp(coef)"].round(3).values,
        "CI_low": s["exp(coef) lower 95%"].round(3).values,
        "CI_high": s["exp(coef) upper 95%"].round(3).values,
        "p": s["p"].values,
    })
    out.to_csv(out_csv, index=False)
    X = sub[list(features)].astype(float)
    vif = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [round(variance_inflation_factor(X.values, i), 2)
                for i in range(X.shape[1])],
    })
    vif.to_csv(out_csv.replace(".csv", "_vif.csv"), index=False)
    try:
        ph = proportional_hazard_test(cph, sub, time_transform="rank")
        sch = ph.summary.reset_index()
        sch.columns = [str(c) for c in sch.columns]
        sch.to_csv(out_csv.replace(".csv", "_schoenfeld.csv"), index=False)
    except Exception as e:
        print(f"  Schoenfeld test failed: {e}")
        sch = pd.DataFrame()
    print(f"Saved {out_csv}, VIF, Schoenfeld")
    return out, vif, sch
