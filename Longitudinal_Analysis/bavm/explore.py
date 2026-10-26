"""Exploratory analysis of how the prior-rupture hazard ratio changes over follow-up."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, NelsonAalenFitter
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import chi2

from .episodes import fit_ctv, split
from .plotting import BLUE, HOPKINS_RED, INK, MUTED, style

MIN_EVENTS_CELL = 3


def _smooth(x, y, frac=0.6, n_out=200):
    """Local-linear smoother with tricube weights."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    grid = np.linspace(x.min(), x.max(), n_out)
    k = max(int(frac * len(x)), 5)
    out = np.full(n_out, np.nan)
    for j, g in enumerate(grid):
        d = np.abs(x - g)
        h = np.sort(d)[min(k, len(d) - 1)] + 1e-12
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        if w.sum() < 1e-8:
            continue
        W = np.sqrt(w)
        A = np.c_[np.ones_like(x), x - g] * W[:, None]
        coef, *_ = np.linalg.lstsq(A, y * W, rcond=None)
        out[j] = coef[0]
    return grid, out


def _interval_labels(edges):
    return [f"{a:g}-{b:g} yr" if np.isfinite(b) else f">{a:g} yr"
            for a, b in zip(edges[:-1], edges[1:])]


def piecewise_hr(df, prior, others, cuts, time_col, event_col):
    """Adjusted prior-rupture HR in each follow-up interval."""
    ep = split(df, [prior] + others, cuts, time_col, event_col)
    edges = np.r_[0.0, np.sort(cuts), np.inf]
    terms = []
    for k in range(len(edges) - 1):
        c = f"prior_p{k}"
        ep[c] = ep[prior] * (ep["interval"] == k)
        terms.append(c)
    s = fit_ctv(ep, others + terms).summary
    rows = []
    for k, (c, lab) in enumerate(zip(terms, _interval_labels(edges))):
        cell = ep[ep["interval"] == k]
        ev1 = int(cell.loc[cell[prior] == 1, "_ev"].sum())
        ev0 = int(cell.loc[cell[prior] == 0, "_ev"].sum())
        rows.append({
            "Interval": lab, "start": edges[k], "stop": edges[k + 1],
            "At_risk_prior": int((cell[prior] == 1).sum()),
            "At_risk_no_prior": int((cell[prior] == 0).sum()),
            "Events_prior": ev1, "Events_no_prior": ev0,
            "HR": s.loc[c, "exp(coef)"],
            "HR_lo": s.loc[c, "exp(coef) lower 95%"],
            "HR_hi": s.loc[c, "exp(coef) upper 95%"],
            "p": s.loc[c, "p"],
            "sparse": ev1 < MIN_EVENTS_CELL or ev0 < MIN_EVENTS_CELL})
    return pd.DataFrame(rows)


def logtime_hr(df, prior, others, time_col, event_col, t_grid):
    """Smooth HR(t) = exp(b0 + b1 log t) from a prior x log(t) interaction."""
    ev_times = np.unique(df.loc[df[event_col] == 1, time_col].to_numpy(float))
    ep = split(df, [prior] + others, ev_times[:-1], time_col, event_col)
    ep["prior_logt"] = ep[prior] * np.log(ep["stop"])
    ctv = fit_ctv(ep, others + [prior, "prior_logt"])
    b = ctv.params_[[prior, "prior_logt"]].to_numpy()
    names = list(ctv.params_.index)
    ix = [names.index(prior), names.index("prior_logt")]
    V = np.asarray(ctv.variance_matrix_, float)[np.ix_(ix, ix)]
    lt = np.log(t_grid)
    lp = b[0] + b[1] * lt
    se = np.sqrt(V[0, 0] + lt ** 2 * V[1, 1] + 2 * lt * V[0, 1])
    curve = pd.DataFrame({"time_yr": t_grid, "HR": np.exp(lp),
                          "HR_lo": np.exp(lp - 1.96 * se),
                          "HR_hi": np.exp(lp + 1.96 * se)})
    return curve, b, float(ctv.summary.loc["prior_logt", "p"])


def schoenfeld_beta(df, prior, others, time_col, event_col):
    """beta(t) from scaled Schoenfeld residuals."""
    d = df[[time_col, event_col, prior] + others].astype(float).reset_index(drop=True)
    cph = CoxPHFitter().fit(d, time_col, event_col)
    r = cph.compute_residuals(d, kind="scaled_schoenfeld")
    beta_t = r[prior].to_numpy() + cph.params_[prior]
    times = d.loc[r.index, time_col].to_numpy(float)
    return times, beta_t, float(cph.params_[prior])


def smoothed_hazards(df, prior, time_col, event_col, bandwidth):
    out = {}
    for g in (0, 1):
        m = df[prior] == g
        naf = NelsonAalenFitter().fit(df.loc[m, time_col], df.loc[m, event_col])
        h = naf.smoothed_hazard_(bandwidth)
        out[g] = (h.index.to_numpy(float), 100 * h.iloc[:, 0].to_numpy(float))
    return out


def cutpoint_scan(df, prior, others, grid, time_col, event_col):
    """Likelihood-ratio statistic of an early/late split at each candidate cut."""
    covs = [prior] + others
    ll0 = fit_ctv(split(df, covs, [], time_col, event_col), covs).log_likelihood_
    rows = []
    for c in grid:
        ep = split(df, covs, [c], time_col, event_col)
        ep["prior_early"] = ep[prior] * (ep["interval"] == 0)
        ep["prior_late"] = ep[prior] * (ep["interval"] == 1)
        ev_e = int(ep.loc[(ep["interval"] == 0) & (ep[prior] == 1), "_ev"].sum())
        ev_l = int(ep.loc[(ep["interval"] == 1) & (ep[prior] == 1), "_ev"].sum())
        ev0_e = int(ep.loc[(ep["interval"] == 0) & (ep[prior] == 0), "_ev"].sum())
        ev0_l = int(ep.loc[(ep["interval"] == 1) & (ep[prior] == 0), "_ev"].sum())
        if min(ev_e, ev_l, ev0_e, ev0_l) < MIN_EVENTS_CELL:
            continue
        ctv = fit_ctv(ep, others + ["prior_early", "prior_late"])
        s = ctv.summary
        lr = 2 * (ctv.log_likelihood_ - ll0)
        rows.append({
            "cut_yr": c, "LR_chi2": lr, "p_naive": chi2.sf(lr, 1),
            "HR_early": s.loc["prior_early", "exp(coef)"],
            "HR_early_lo": s.loc["prior_early", "exp(coef) lower 95%"],
            "HR_early_hi": s.loc["prior_early", "exp(coef) upper 95%"],
            "HR_late": s.loc["prior_late", "exp(coef)"],
            "HR_late_lo": s.loc["prior_late", "exp(coef) lower 95%"],
            "HR_late_hi": s.loc["prior_late", "exp(coef) upper 95%"],
            "Events_prior_early": ev_e, "Events_prior_late": ev_l,
            "Events_no_prior_early": ev0_e, "Events_no_prior_late": ev0_l})
    return pd.DataFrame(rows)


def _draw_hr(ax, curve, pw, hr_ph, p_logt, xmax, best=None,
             xlabel="Years since study entry"):
    ax.fill_between(curve["time_yr"], curve["HR_lo"], curve["HR_hi"],
                    color=HOPKINS_RED, alpha=0.15, lw=0)
    ax.plot(curve["time_yr"], curve["HR"], color=HOPKINS_RED, lw=2.6,
            label=f"Prior rupture × log(t) model (p = {p_logt:.3f})")
    for _, r in pw.iterrows():
        a, bnd = r["start"], min(r["stop"], xmax)
        if a >= xmax or not np.isfinite(r["HR_hi"]) or r["HR"] > 1e3:
            continue
        mid = (a + bnd) / 2
        ax.hlines(r["HR"], a, bnd, color=INK, lw=2, ls=":" if r["sparse"] else "-")
        ax.errorbar(mid, r["HR"], yerr=[[r["HR"] - r["HR_lo"]], [r["HR_hi"] - r["HR"]]],
                    fmt="o", color=INK, ms=4, capsize=3, lw=1)
    ax.plot([], [], color=INK, lw=2, marker="o", ms=4, label="Piecewise HR (95% CI)")
    ax.axhline(hr_ph, color=MUTED, ls="--", lw=1.2,
               label=f"Constant HR (PH model) = {hr_ph:.2f}")
    ax.axhline(1, color="#BBBBBB", lw=1)
    ax.set_yscale("log")
    ok = pw[np.isfinite(pw["HR_hi"]) & (pw["HR"] < 1e3)]
    lo = min(curve["HR_lo"].min(), ok["HR_lo"].min() if len(ok) else 1, 0.5)
    hi = max(curve["HR_hi"].max(), ok["HR_hi"].max() if len(ok) else 1, 2)
    ax.set_ylim(lo / 1.5, min(hi * 1.5, 100))
    y0, y1 = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator(
        [t for t in (0.25, 0.5, 1, 2, 5, 10, 20, 50) if y0 <= t <= y1]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, xmax)
    if best is not None:
        ax.axvline(best["cut_yr"], color=HOPKINS_RED, ls=":", lw=1.3)
        ax.text(best["cut_yr"], ax.get_ylim()[1] / 1.15,
                f"  cut = {best['cut_yr']:g} yr\n"
                f"  HR {best['HR_early']:.2f} before, {best['HR_late']:.2f} after",
                va="top", ha="left", fontsize=9, color=HOPKINS_RED)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Hazard ratio for prior rupture (log scale)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.15, which="both")


def run_time_varying_prior(df, features, prior_col="Ever_Ruptured",
                           time_col="time", event_col="event",
                           piecewise_cuts=(0.5, 1, 2, 5),
                           scan_grid=np.arange(0.25, 5.01, 0.25),
                           bandwidth=1.0, max_years=10.0,
                           fig_name="FIGURE_prior_rupture_time_varying"):
    """Piecewise HR, log-time HR, Schoenfeld beta(t), hazards and cut-point scan."""
    others = [f for f in features if f != prior_col]
    d = df[[time_col, event_col, prior_col] + others].dropna().astype(float)
    d = d.reset_index(drop=True)
    ev_t = d.loc[d[event_col] == 1, time_col]
    xmax = min(float(np.percentile(ev_t, 95)), float(max_years))
    t_grid = np.linspace(max(0.1, ev_t.min()), xmax, 200)

    print("\nTIME-VARYING EFFECT OF PRIOR RUPTURE")
    print(f"  adjusted for: {others}")

    # Piecewise HR
    pw = piecewise_hr(d, prior_col, others, list(piecewise_cuts), time_col, event_col)
    pw.round(4).to_csv("tv_prior_piecewise.csv", index=False)
    print("\n  Piecewise HR by follow-up interval:")
    print(pw[["Interval", "Events_prior", "Events_no_prior", "HR", "HR_lo",
              "HR_hi", "p", "sparse"]].round(3).to_string(index=False))
    if pw["sparse"].any():
        print(f"  NOTE: intervals flagged 'sparse' have <{MIN_EVENTS_CELL} events "
              f"in a group; their HRs are unstable.")

    # Log-time interaction
    curve, b, p_logt = logtime_hr(d, prior_col, others, time_col, event_col, t_grid)
    curve.round(4).to_csv("tv_prior_logtime.csv", index=False)
    print(f"\n  prior x log(t) model: HR(t) = exp({b[0]:.3f} + {b[1]:.3f}*log t), "
          f"interaction p = {p_logt:.4f}")
    for tt in (0.5, 1, 2, 5):
        if tt <= xmax:
            i = np.argmin(np.abs(curve["time_yr"] - tt))
            print(f"    HR at {tt:g} yr: {curve['HR'][i]:.2f} "
                  f"[{curve['HR_lo'][i]:.2f}, {curve['HR_hi'][i]:.2f}]")

    st, sb, beta_ph = schoenfeld_beta(d, prior_col, others, time_col, event_col)
    hr_ph = float(np.exp(beta_ph))
    haz = smoothed_hazards(d, prior_col, time_col, event_col, bandwidth)

    # Cut-point scan
    scan = cutpoint_scan(d, prior_col, others, scan_grid, time_col, event_col)
    scan.round(4).to_csv("tv_prior_cutpoint_scan.csv", index=False)
    best = scan.loc[scan["LR_chi2"].idxmax()] if len(scan) else None
    if best is not None:
        two = pd.DataFrame([{
            "cut_yr": best["cut_yr"], "Period": f"0-{best['cut_yr']:g} yr",
            "HR": best["HR_early"], "HR_lo": best["HR_early_lo"],
            "HR_hi": best["HR_early_hi"], "Events_prior": best["Events_prior_early"]}, {
            "cut_yr": best["cut_yr"], "Period": f">{best['cut_yr']:g} yr",
            "HR": best["HR_late"], "HR_lo": best["HR_late_lo"],
            "HR_hi": best["HR_late_hi"], "Events_prior": best["Events_prior_late"]}])
        two.round(4).to_csv("tv_prior_two_period.csv", index=False)
        print(f"\n  Best-fitting cut-point: {best['cut_yr']:g} yr "
              f"(LR chi2 {best['LR_chi2']:.2f} vs. constant HR; naive p "
              f"{best['p_naive']:.4f}, anti-conservative because the cut "
              f"was selected from the data)")
        print(two.round(3).to_string(index=False))
    print(f"  Constant-HR (PH) model: HR = {hr_ph:.2f}")

    # Four-panel figure
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(14, 11))
    _draw_hr(axA, curve, pw, hr_ph, p_logt, xmax)

    keep = st <= xmax
    axB.scatter(st[keep], sb[keep], s=12, color=MUTED, alpha=0.5, lw=0)
    gx, gy = _smooth(st[keep], sb[keep])
    axB.plot(gx, gy, color=HOPKINS_RED, lw=2.6, label="Smoothed β(t)")
    axB.axhline(beta_ph, color=MUTED, ls="--", lw=1.2,
                label=f"Constant β (PH) = {beta_ph:.2f}")
    axB.axhline(0, color="#BBBBBB", lw=1)
    lo, hi = np.percentile(sb[keep], [3, 97])
    pad = 0.15 * (hi - lo)
    axB.set_ylim(lo - pad, hi + pad)
    axB.set_xlim(0, xmax)
    axB.set_xlabel("Years since study entry", fontsize=11)
    axB.set_ylabel("β(t) for prior rupture (log hazard ratio)", fontsize=11)
    axB.legend(fontsize=8.5, frameon=False, loc="upper right")
    axB.grid(True, alpha=0.15)

    for g, col, lab in [(1, HOPKINS_RED, "Prior rupture"), (0, BLUE, "No prior rupture")]:
        tx, hy = haz[g]
        m = tx <= xmax
        axC.plot(tx[m], hy[m], color=col, lw=2.4, label=lab)
    axC.set_xlim(0, xmax)
    axC.set_ylim(bottom=0)
    axC.set_xlabel("Years since study entry", fontsize=11)
    axC.set_ylabel("Smoothed hazard of rupture\n(per 100 person-years)", fontsize=11)
    axC.legend(fontsize=9, frameon=False, loc="upper right")
    axC.text(0.98, 0.70, f"Kernel bandwidth {bandwidth:g} yr", ha="right",
             transform=axC.transAxes, fontsize=8, color=MUTED, style="italic")
    axC.grid(True, alpha=0.15)

    if len(scan):
        axD.plot(scan["cut_yr"], scan["LR_chi2"], color=INK, lw=2, marker="o", ms=4)
        axD.axvline(best["cut_yr"], color=HOPKINS_RED, ls="--", lw=1.4)
        axD.axhline(chi2.ppf(0.95, 1), color=MUTED, ls=":", lw=1,
                    label="χ² = 3.84 (naive p = 0.05)")
        axD.text(best["cut_yr"], axD.get_ylim()[1] * 0.97,
                 f"  best cut = {best['cut_yr']:g} yr\n"
                 f"  HR early {best['HR_early']:.2f} "
                 f"[{best['HR_early_lo']:.2f}-{best['HR_early_hi']:.2f}]\n"
                 f"  HR late  {best['HR_late']:.2f} "
                 f"[{best['HR_late_lo']:.2f}-{best['HR_late_hi']:.2f}]",
                 va="top", ha="left", fontsize=8.5, color=HOPKINS_RED,
                 family="monospace")
        axD.legend(fontsize=8.5, frameon=False, loc="center right")
    axD.set_xlabel("Candidate cut-point (years since study entry)", fontsize=11)
    axD.set_ylabel("Likelihood-ratio χ²\n(early/late HR vs. constant HR)",
                   fontsize=11)
    axD.grid(True, alpha=0.15)

    for ax, letter, title in [
            (axA, "A", "Hazard ratio for prior rupture over time"),
            (axB, "B", "Scaled Schoenfeld residuals (β(t) diagnostic)"),
            (axC, "C", "Hazard of rupture by prior-rupture status"),
            (axD, "D", "Where does the effect change? (cut-point scan)")]:
        ax.set_title(f"{letter}. {title}", fontsize=12, fontweight="bold", loc="left", pad=10)
        style(ax)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fig_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_name}.png / .pdf")

    # Single-panel HR(t)
    fs, axs = plt.subplots(figsize=(8, 6))
    _draw_hr(axs, curve, pw, hr_ph, p_logt, xmax, best=best, xlabel="Years from diagnosis")
    axs.set_title("Hazard ratio for prior rupture over time", fontsize=12.5,
                  fontweight="bold", loc="left", pad=10)
    style(axs)
    fs.tight_layout()
    fs.savefig(f"{fig_name}_single.png", dpi=300, bbox_inches="tight")
    fs.savefig(f"{fig_name}_single.pdf", bbox_inches="tight")
    plt.close(fs)
    print(f"  Saved {fig_name}_single.png / .pdf")

    return {"piecewise": pw, "logtime_curve": curve, "logtime_p": p_logt,
            "scan": scan, "best_cut": None if best is None else float(best["cut_yr"]),
            "hr_ph": hr_ph}
