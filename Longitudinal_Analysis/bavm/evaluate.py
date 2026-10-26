"""Discrimination and rate summaries: bootstrap C-index, td-AUC, log-rank and interval rates."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sksurv.metrics import cumulative_dynamic_auc

import config as cfg
from .model import cindex_tv
from .stats import annual_rate


def boot_c(T, E, le, ll, cut, mask=None, seed=None, n_boot=None):
    """Time-varying C-index with bootstrap 95% CI, optionally in a subset."""
    seed = cfg.SEED if seed is None else seed
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    n = len(T)
    idx0 = np.arange(n) if mask is None else np.flatnonzero(mask)
    T_, E_ = T[idx0], E[idx0]
    le_, ll_ = np.asarray(le, float)[idx0], np.asarray(ll, float)[idx0]
    cut_ = np.broadcast_to(np.asarray(cut, float), (n,))[idx0]
    c = cindex_tv(T_, E_, le_, ll_, cut_)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        b = rng.choice(len(idx0), len(idx0), replace=True)
        if E_[b].sum() >= 5:
            bs.append(cindex_tv(T_[b], E_[b], le_[b], ll_[b], cut_[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return c, lo, hi, len(idx0), int(E_.sum())


def boot_auc(y_surv, E, est, times, seed):
    """Bootstrap 95% band for the time-dependent AUC."""
    n = len(E)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(cfg.N_BOOT_TDAUC):
        idx = rng.choice(n, n, replace=True)
        if E[idx].sum() < 5:
            continue
        try:
            a, _ = cumulative_dynamic_auc(y_surv, y_surv[idx], est[idx], times)
            out.append(a)
        except Exception:
            continue
    if not out:
        return None, None
    out = np.vstack(out)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def logrank_counting(ep, group_col, groups):
    """K-group log-rank test on (start, stop] data, so patients can change group."""
    st, sp = ep["start"].to_numpy(float), ep["stop"].to_numpy(float)
    evv, g = ep["_ev"].to_numpy(int), ep[group_col].to_numpy()
    K = len(groups)
    O_E = np.zeros(K)
    V = np.zeros((K, K))
    for t in np.unique(sp[evv == 1]):
        at = (st < t) & (sp >= t)
        dk = (sp == t) & (evv == 1)
        n_t, d_t = at.sum(), dk.sum()
        if n_t < 2:
            continue
        n_k = np.array([(at & (g == gg)).sum() for gg in groups], float)
        d_k = np.array([(dk & (g == gg)).sum() for gg in groups], float)
        O_E += d_k - d_t * n_k / n_t
        f = d_t * (n_t - d_t) / (n_t ** 2 * (n_t - 1))
        V += f * (np.diag(n_k) * n_t - np.outer(n_k, n_k))
    stat = float(O_E[:-1] @ np.linalg.pinv(V[:-1, :-1]) @ O_E[:-1])
    return stat, float(chi2.sf(stat, K - 1))


def interval_rates(d, groups, cuts):
    """Rupture rate per 100 person-years in each follow-up interval by prior-rupture group."""
    edges = np.r_[0.0, np.sort(np.asarray(cuts, float)), np.inf]
    labels = [f"{a:g}-{b:g} yr" if np.isfinite(b) else f">{a:g} yr"
              for a, b in zip(edges[:-1], edges[1:])]
    T, E = d["time"].to_numpy(float), d["event"].to_numpy(int)
    rows = []
    for g, lab, _ in groups:
        m = d[cfg.PRIOR].to_numpy() == g
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            pt = np.clip(np.minimum(T[m], b) - a, 0, None)
            ev = (E[m] == 1) & (T[m] > a) & (T[m] <= b)
            r, lo, hi = annual_rate(pt[pt > 0], ev[pt > 0].astype(int))
            rows.append({"Group": lab, "prior": g, "Interval": labels[k],
                         "Person_years": float(pt.sum()), "Events": int(ev.sum()),
                         "rate_pct": 100 * r, "lo_pct": 100 * lo, "hi_pct": 100 * hi})
    return pd.DataFrame(rows), labels