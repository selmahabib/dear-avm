"""Figure panels and layouts for the time-varying analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from mpl_toolkits.axes_grid1 import make_axes_locatable

TIER_COLORS = {"Low": "#3E8E5A", "Medium": "#D9922B", "High": "#B5253A"}
TIER_ORDER = ["Low", "Medium", "High"]
NAVY = "#0E2A47"
HOPKINS_RED = "#9D2235"
SMOOTH_COLOR = HOPKINS_RED
BLUE = "#2E6FB7"
GOLD = "#C28E0E"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
SLATE = "#7f8c8d"
GREY = "#8C8C8C"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, stem, pdf=True):
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def title(ax, letter, text, size=12.5):
    ax.set_title(f"{letter}. {text}" if letter else text, fontsize=size,
                 fontweight="bold", loc="left", pad=10)


# Shared panels

def panel_rate(ax, rate_df):
    """Hemorrhage incidence per 100 person-years by tier."""
    xp = np.arange(len(rate_df))
    ax.bar(xp, rate_df["rate_pct"], 0.55,
           color=[TIER_COLORS[t] for t in rate_df["Tier"]], alpha=0.85)
    ax.errorbar(xp, rate_df["rate_pct"],
                yerr=[rate_df["rate_pct"] - rate_df["lo_pct"],
                      rate_df["hi_pct"] - rate_df["rate_pct"]],
                fmt="none", ecolor="#3A3A3A", capsize=4, lw=1.2)
    for x, r in zip(xp, rate_df.itertuples()):
        ax.text(x, r.hi_pct + 0.4, f"{r.rate_pct:.1f}\n(n={r.N}, ev={r.Events})",
                ha="center", fontsize=8.5)
    ax.set_xticks(xp)
    ax.set_xticklabels([f"{t}\n({lbl})" for t, lbl in
                        zip(rate_df["Tier"], rate_df["Score_range"])], fontsize=10)
    ax.set_ylabel("Hemorrhages per 100 person-years", fontsize=11)
    ax.set_xlabel("Integer-score tier", fontsize=11)
    ax.set_ylim(0, max(rate_df["hi_pct"]) * 1.35)
    ax.grid(ls=":", alpha=0.3, axis="y")


def panel_calibration(ax, cdf, slope, intercept, horizon, cap=None, tick_step=0.02,
                      cox_slope=None, y_cap=None, connect=True, box=True,
                      fit_label="Quartile"):
    """Grouped KM observed vs predicted risk with 95% CIs."""
    if cap is None:
        cap = float(np.ceil(cdf["pred"].max() * 1.20 / tick_step) * tick_step)
    if y_cap is None:
        y_cap = max(cap, cdf["obs_hi"].max() * 1.05)
        y_cap = float(np.ceil(y_cap / tick_step) * tick_step)
    lim = max(cap, y_cap)
    ax.plot([0, lim], [0, lim], "--", color=SLATE, lw=1.3, zorder=1)
    if connect:
        ax.plot(cdf["pred"], cdf["obs"], "-", color="#7FB3D8", lw=1.6, zorder=2)
    half = cap * 0.012
    for xv, lov, hiv in zip(cdf["pred"], cdf["obs_lo"], cdf["obs_hi"]):
        ax.vlines(xv, lov, hiv, color=NAVY, lw=1.0, alpha=0.5, zorder=3)
        ax.hlines([lov, hiv], xv - half, xv + half, color=NAVY, lw=1.0, alpha=0.5, zorder=3)
    ax.plot(cdf["pred"], cdf["obs"], "o", color=NAVY, markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    xt = np.arange(0, cap + 1e-9, tick_step)
    yt = np.arange(0, y_cap + 1e-9, tick_step)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{t * 100:.0f}" for t in xt])
    ax.set_yticklabels([f"{t * 100:.0f}" for t in yt])
    ax.set_xlim(0, cap)
    ax.set_ylim(0, y_cap)
    ax.set_xlabel(f"Predicted {horizon:.0f}-year rupture risk (%)", fontsize=11)
    ax.set_ylabel(f"Observed {horizon:.0f}-year rupture risk (Kaplan-Meier, %)", fontsize=11)
    if box:
        txt = ""
        if cox_slope is not None:
            s, lo, hi = cox_slope
            txt += f"Calibration slope {s:.2f} (95% CI {lo:.2f}-{hi:.2f})\n"
        txt += f"{fit_label} linear fit: slope {slope:.2f}, intercept {intercept:.3f}"
        ax.text(0.04, 0.95, txt, transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))
    ax.grid(ls=":", alpha=0.3)


def overlay_smooth(ax, sm):
    if sm["lo"] is not None:
        ax.fill_between(sm["grid"], sm["lo"], sm["hi"], color=SMOOTH_COLOR,
                        alpha=0.13, lw=0, zorder=1.5)
    ax.plot(sm["grid"], sm["obs"], color=SMOOTH_COLOR, lw=2.2, zorder=2.5)


def add_hist_strip(ax, pred, T, E, h, cap):
    """Strip of predicted risks under a calibration plot: events up, others down."""
    raw = np.asarray(pred, float)
    n_over = int((raw >= cap).sum())
    pred = np.clip(raw, 0, cap * 0.999)
    ev = (E == 1) & (T <= h)
    binw = cap / 50
    bins = np.arange(0, cap + binw, binw)
    c_ev, _ = np.histogram(pred[ev], bins)
    c_ne, _ = np.histogram(pred[~ev], bins)
    mid = (bins[:-1] + bins[1:]) / 2
    sax = make_axes_locatable(ax).append_axes("bottom", size="16%", pad=0.08, sharex=ax)
    if c_ev.max() > 0:
        sax.bar(mid, c_ev / c_ev.max(), width=binw * 0.9, color=SMOOTH_COLOR, lw=0)
    if c_ne.max() > 0:
        sax.bar(mid, -c_ne / c_ne.max(), width=binw * 0.9, color="#9AA3AD", lw=0)
    sax.axhline(0, color="#555555", lw=0.6)
    sax.set_ylim(-1.15, 1.15)
    sax.set_yticks([])
    sax.text(0.99, 0.98, f"Rupture by {h:g} yr (n={int(ev.sum())})",
             transform=sax.transAxes, ha="right", va="top", fontsize=7.5,
             color=SMOOTH_COLOR, bbox=dict(facecolor="white", edgecolor="none",
                                           alpha=0.8, pad=0.5))
    sax.text(0.99, 0.02, f"No rupture by {h:g} yr (n={int((~ev).sum())})",
             transform=sax.transAxes, ha="right", va="bottom", fontsize=7.5,
             color="#6B7480", bbox=dict(facecolor="white", edgecolor="none",
                                        alpha=0.8, pad=0.5))
    if n_over:
        sax.text(0.01, 0.98, f"{n_over} patients with predicted risk "
                 f"≥{100 * cap:g}% shown in last bin", transform=sax.transAxes,
                 ha="left", va="top", fontsize=7, color="#6B7480")
    xl = ax.get_xlabel()
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)
    sax.set_xlabel(xl, fontsize=11)
    for sp in ("top", "right", "left"):
        sax.spines[sp].set_visible(False)
    return sax


def calibration_panel(ax, cdf, sl, ic, cm, sm, h, x_max, y_max, tick, T, E, pred,
                      show_band, show_hist, legend_fs=8.5, at_risk_line=False):
    """Main-figure calibration panel: smoothed curve, quartile points, O/E and slope."""
    panel_calibration(ax, cdf, sl, ic, h, x_max, tick_step=tick, y_cap=y_max,
                      connect=False, box=False)
    if show_band and sm["lo"] is not None:
        ax.fill_between(sm["grid"], sm["lo"], sm["hi"], color=NAVY, alpha=0.12,
                        lw=0, zorder=1.5)
    ax.plot(sm["grid"], sm["obs"], "-", color=NAVY, lw=2.2, zorder=2.5,
            label="Smoothed calibration curve" + (" (95% CI)" if show_band else ""))
    ax.plot([], [], "--", color=MUTED, lw=1.3, label="Perfect calibration")
    ax.plot([], [], "o", color=NAVY, markersize=7,
            label="Quartiles of predicted risk (KM, 95% CI)")
    ax.legend(loc="lower right", fontsize=legend_fs, frameon=True, framealpha=0.9)
    txt = (f"O/E {cm['OE']:.2f} (95% CI {cm['OE_lo']:.2f}-{cm['OE_hi']:.2f})\n"
           f"Calibration slope {cm['slope']:.2f} "
           f"(95% CI {cm['slope_lo']:.2f}-{cm['slope_hi']:.2f})\n"
           f"ICI {100 * sm['ICI']:.1f}%, E90 {100 * sm['E90']:.1f}%")
    if at_risk_line:
        txt += (f"\nAt risk at {h:g} yr: {int((T >= h).sum())}; "
                f"ruptures by {h:g} yr: {int(((T <= h) & (E == 1)).sum())}")
    ax.text(0.04, 0.95, txt, transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    style(ax)
    if show_hist:
        add_hist_strip(ax, pred, T, E, h, x_max)


# Prior vs no prior rupture

def figure_prior_vs_no_prior(d, rt, labels, groups, hr_by_int, cuts, prior_col,
                             t_max=10.0, fname="FIGURE_prior_vs_no_prior_risk"):
    """Cumulative incidence and interval rates for prior vs no prior rupture."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={"wspace": 0.25})
    fitters = []
    for g, lab, col in groups:
        m = d[prior_col] == g
        kmf = KaplanMeierFitter().fit(d.loc[m, "time"], d.loc[m, "event"],
                                      label=f"{lab} (n={int(m.sum())})")
        ci = kmf.confidence_interval_survival_function_
        t_ = kmf.survival_function_.index.to_numpy()
        axA.step(t_, 100 * (1 - kmf.survival_function_.iloc[:, 0]), where="post",
                 color=col, lw=2.2, label=kmf._label)
        axA.fill_between(ci.index, 100 * (1 - ci.iloc[:, 0]), 100 * (1 - ci.iloc[:, 1]),
                         step="post", color=col, alpha=0.12, lw=0)
        fitters.append(kmf)
    for c in cuts:
        if c < t_max:
            axA.axvline(c, color=SLATE, ls="--", lw=1)
    axA.set_xlim(0, t_max)
    axA.set_ylim(bottom=0)
    axA.set_xticks(np.arange(0, t_max + 1e-9, 2))
    axA.set_xlabel("Years from diagnosis", fontsize=11)
    axA.set_ylabel("Cumulative incidence of rupture (%)", fontsize=11)
    axA.legend(loc="upper left", fontsize=9.5, frameon=False)
    axA.grid(ls=":", alpha=0.3)
    add_at_risk_counts(*fitters, ax=axA, rows_to_show=["At risk"])

    xp, w = np.arange(len(labels)), 0.36
    for j, (g, lab, col) in enumerate(groups):
        sub = rt[rt["prior"] == g]
        x = xp + (j - 0.5) * w
        axB.bar(x, sub["rate_pct"], w, color=col, alpha=0.85, label=lab)
        axB.errorbar(x, sub["rate_pct"],
                     yerr=[sub["rate_pct"] - sub["lo_pct"], sub["hi_pct"] - sub["rate_pct"]],
                     fmt="none", ecolor="#3A3A3A", capsize=3, lw=1.1)
        for xi, r in zip(x, sub.itertuples()):
            axB.text(xi, r.hi_pct + 0.25, f"{r.rate_pct:.1f}\n(ev={r.Events})",
                     ha="center", fontsize=8.5)
    ymax = rt["hi_pct"].max() * 1.3
    for k, lab in enumerate(labels):
        hr = hr_by_int.get(lab, np.nan)
        if np.isfinite(hr):
            axB.text(k, ymax * 0.93, f"adj. HR {hr:.2f}", ha="center", fontsize=9.5,
                     fontweight="bold", color=INK)
    axB.set_xticks(xp)
    axB.set_xticklabels(labels, fontsize=10.5)
    axB.set_ylim(0, ymax)
    axB.set_xlabel("Follow-up interval after diagnosis", fontsize=11)
    axB.set_ylabel("Ruptures per 100 person-years", fontsize=11)
    axB.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=9.5, frameon=False)
    axB.grid(ls=":", alpha=0.3, axis="y")

    title(axA, "A", "Cumulative incidence of rupture")
    title(axB, "B", "Rupture rate by follow-up interval")
    style(axA)
    style(axB)
    save(fig, fname)


# Time-varying model

def figure_tv_model(u_tv, cal, p_wald, cut, horizon, fname="FIGURE_tv_model"):
    """Hazard ratios and calibration of the time-varying model."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    # Forest plot
    rows = [("age", "Age (per 10 yr)", 10), ("deep", "Exclusive deep location", 1),
            ("prior_early", f"Prior rupture, 0-{cut:g} yr", 1),
            ("prior_late", f"Prior rupture, >{cut:g} yr", 1)]
    for y, (c, lab, k) in enumerate(rows):
        s = u_tv["ctv"].summary.loc[c]
        hr = np.exp(k * s["coef"])
        lo, hi = np.exp(k * s["coef lower 95%"]), np.exp(k * s["coef upper 95%"])
        axA.errorbar(hr, -y, xerr=[[hr - lo], [hi - hr]], fmt="o", color=HOPKINS_RED,
                     ms=6, capsize=3, lw=1.6)
        axA.text(hi * 1.08, -y, f"{hr:.2f} [{lo:.2f}-{hi:.2f}]", va="center", fontsize=8.5,
                 color=HOPKINS_RED)
    axA.axvline(1, color="#BBBBBB", lw=1)
    axA.set_xscale("log")
    axA.set_yticks([-y for y in range(len(rows))])
    axA.set_yticklabels([lab for _, lab, _ in rows], fontsize=10)
    axA.set_ylim(-len(rows) + 0.5, 0.5)
    axA.set_xlabel("Hazard ratio (95% CI, log scale)", fontsize=11)
    axA.text(0.02, 0.02, f"Early vs late HR: Wald p = {p_wald:.3f}",
             transform=axA.transAxes, fontsize=9, color=MUTED, style="italic")
    axA.grid(True, axis="x", alpha=0.15, which="both")

    # Calibration
    cap = max(cal["pred"].max(), cal["obs_hi"].max()) * 1.1
    axB.plot([0, cap], [0, cap], ":", color="#BBBBBB", lw=1.2)
    axB.errorbar(cal["pred"], cal["obs"], yerr=[cal["obs"] - cal["obs_lo"], cal["obs_hi"] - cal["obs"]],
                 fmt="o-", color=HOPKINS_RED, ms=7, capsize=3, lw=1.6)
    axB.set_xlim(0, cap)
    axB.set_ylim(0, cap)
    axB.set_xlabel(f"Predicted {horizon:g}-yr rupture risk (out-of-fold)", fontsize=11)
    axB.set_ylabel(f"Observed {horizon:g}-yr risk (Kaplan-Meier)", fontsize=11)
    axB.grid(True, alpha=0.15)

    title(axA, "A", "Hazard ratios, time-varying model", size=12)
    title(axB, "B", f"Calibration at {horizon:g} years", size=12)
    style(axA)
    style(axB)
    fig.tight_layout()
    save(fig, fname)


# Main longitudinal figure

def panel_km_tv(ax, ep_t, tier_labels, info, t_max):
    """KM by time-updated score tier (prior-rupture patients move tier at the cut)."""
    fitters = []
    for t in TIER_ORDER:
        m = ep_t["tier"] == t
        if not m.any():
            continue
        kmf = KaplanMeierFitter().fit(
            ep_t.loc[m, "stop"], ep_t.loc[m, "_ev"], entry=ep_t.loc[m, "start"],
            label=f"{tier_labels[t]} (n={ep_t.loc[m, '_id'].nunique()})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=TIER_COLORS[t],
                                   linewidth=2.0, ci_alpha=0.10)
        fitters.append(kmf)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(np.arange(0, t_max + 1e-9, 2))
    ax.set_xlabel("Years from diagnosis", fontsize=11)
    ax.set_ylabel("Cumulative rupture-free survival", fontsize=11)
    ax.text(0.03, 0.28, info, transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#BBBBBB",
                      boxstyle="square,pad=0.5"))
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.03, 0.03), fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor="#BBBBBB",
                    borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(0.8)
    ax.grid(ls=":", alpha=0.3)
    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])


def panel_td_auc(ax, times, auc_tv, tv_lo, tv_hi, auc_r2, r2_lo, r2_hi, cut):
    ax.plot(times, auc_tv, "o-", color=NAVY, lw=2, markersize=5,
            label=f"Time-varying model (prior rupture split at {cut:g} yr)")
    if tv_lo is not None:
        ax.fill_between(times, tv_lo, tv_hi, color=NAVY, alpha=0.15)
    ax.plot(times, auc_r2, "s-", color=GOLD, lw=1.6, markersize=4,
            label="R$_2$eD AVM score, refit as Cox model")
    if r2_lo is not None:
        ax.fill_between(times, r2_lo, r2_hi, color=GOLD, alpha=0.12)
    ax.axhline(0.5, ls=":", color="#5B9BD5", lw=1)
    ax.set_xlabel("Time horizon (years)", fontsize=11)
    ax.set_ylabel("Time-dependent AUC", fontsize=11)
    ax.set_ylim(0.4, 0.95)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(ls=":", alpha=0.3)


def figure_longitudinal(draw_auc, draw_cal, draw_km, rate_df, horizon,
                        fname="FIGURE_longitudinal_time_varying"):
    """Four panels: td-AUC, calibration, KM by tier, incidence by tier."""
    fig, ((axA, axB), (axC, axD)) = plt.subplots(
        2, 2, figsize=(16, 14), gridspec_kw={"wspace": 0.22, "hspace": 0.42})
    draw_km(axC)
    panel_rate(axD, rate_df)
    draw_cal(axB)
    draw_auc(axA)
    for ax, letter, text in [
            (axA, "A", "Time-dependent AUC (cross-validated)"),
            (axB, "B", f"Calibration of the time-varying model at {horizon:.0f} years"),
            (axC, "C", "Kaplan-Meier by time-varying integer-score tier"),
            (axD, "D", "Hemorrhage incidence by tier")]:
        title(ax, letter, text)
        style(ax)
    save(fig, fname)


# Calibration across horizons

def figure_calibration_horizons(hz, cal_tabs, smooth, risk_by_h, slope_ci, T, E,
                                stem, what, out, cal_tick, is_score, tiers):
    """Calibration at each horizon (one grid figure plus one file per horizon)."""
    fit_label = "Tier" if tiers else "Quartile"

    def draw(ax, h, row, cdf):
        top = float(max(cdf["pred"].max(), cdf["obs_hi"].max()))
        cap = float(np.ceil(cdf["pred"].max() * 1.2 / cal_tick) * cal_tick)
        ycap = float(np.ceil(max(cap, top * 1.05) / cal_tick) * cal_tick)
        panel_calibration(ax, cdf, row["group_slope"], row["group_intercept"], h,
                          cap=cap, tick_step=cal_tick, cox_slope=slope_ci, y_cap=ycap,
                          fit_label=fit_label if is_score else "Quartile")
        if is_score and tiers:
            for r in cdf.itertuples():
                ax.annotate(r.tier, (r.pred, r.obs), textcoords="offset points",
                            xytext=(8, -4), fontsize=8.5, color=TIER_COLORS[r.tier])
        ax.set_xlabel(f"Predicted {h:g}-year rupture risk (%)", fontsize=11)
        ax.set_ylabel(f"Observed {h:g}-year risk (Kaplan-Meier, %)", fontsize=11)
        overlay_smooth(ax, smooth[h])
        ax.text(0.96, 0.04, f"At risk at {h:g} yr: {int(row['at_risk'])}\n"
                f"Events by {h:g} yr: {int(row['events_by_h'])}\n"
                f"Expected/observed: {row['E_O_ratio']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=SLATE)
        ax.text(0.96, 0.20, f"Smoothed curve (95% CI)\n"
                f"ICI {100 * row['ICI']:.2f}%, E90 {100 * row['E90']:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=SMOOTH_COLOR)
        style(ax)
        add_hist_strip(ax, risk_by_h[h], T, E, h, cap)

    horizons = list(cal_tabs)
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), gridspec_kw={"wspace": 0.28, "hspace": 0.38})
    axes = axes.ravel()
    for i, h in enumerate(horizons):
        row, cdf = hz.iloc[i], cal_tabs[h]
        plural = "s" if h != 1 else ""
        draw(axes[i], h, row, cdf)
        title(axes[i], chr(65 + i), f"Calibration at {h:g} year{plural}")
        f1, a1 = plt.subplots(figsize=(7.5, 7))
        draw(a1, h, row, cdf)
        title(a1, None, f"Calibration of the {what} at {h:g} year{plural}")
        save(f1, f"{stem}_{f'{h:g}'.replace('.', 'p')}yr", pdf=False)
    ax = axes[-1]
    ax.axis("off")
    tbl = [[f"{r.horizon_yr:g}", f"{r.at_risk}", f"{r.events_by_h}",
            f"{100 * r.mean_pred:.1f}", f"{100 * r.KM_obs:.1f}", f"{r.E_O_ratio:.2f}",
            f"{100 * r.ICI:.2f}"] for r in hz.itertuples()]
    t_ = ax.table(cellText=tbl, colLabels=["Horizon\n(yr)", "At\nrisk", "Events\nby t",
                                           "Mean pred\n(%)", "KM obs\n(%)", "E/O", "ICI\n(%)"],
                  loc="center", cellLoc="center")
    t_.auto_set_font_size(False)
    t_.set_fontsize(9)
    t_.scale(1.05, 2.3)
    title(ax, chr(65 + len(horizons)),
          f"Summary, {what} (Cox calibration slope {slope_ci[0]:.2f}, "
          f"95% CI {slope_ci[1]:.2f}-{slope_ci[2]:.2f})", size=12)
    save(fig, out)


# Supplement

def figure_observed_vs_predicted(T, E, groups, td_pred, td_times, path):
    """Observed (KM) vs mean predicted cumulative risk over time by risk tertile."""
    t_grid = np.r_[0.0, td_times]
    t_max = float(min(10.0, td_times.max()))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    rows = []
    for gname in ["Low", "Medium", "High"]:
        m = groups == gname
        col = TIER_COLORS[gname]
        kmf = KaplanMeierFitter().fit(T[m], E[m])
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        keep = sf.index <= t_max
        ax.step(sf.index[keep], 100 * (1 - sf.iloc[:, 0][keep]), where="post",
                color=col, lw=2.0, label=f"{gname} predicted risk: observed (KM), n={int(m.sum())}")
        ax.fill_between(ci.index[keep], 100 * (1 - ci.iloc[:, 1][keep]),
                        100 * (1 - ci.iloc[:, 0][keep]), step="post", color=col,
                        alpha=0.10, lw=0)
        pred_curve = np.r_[0.0, np.nanmean(td_pred[m], axis=0)]
        ax.plot(t_grid, 100 * pred_curve, "--o", color=col, lw=1.6, markersize=4,
                label=f"{gname} predicted risk: mean predicted")
        for k, t in enumerate(td_times):
            rows.append({"group": gname, "time_yr": float(t), "n": int(m.sum()),
                         "at_risk": int((T[m] >= t).sum()),
                         "observed_KM": 1 - float(kmf.survival_function_at_times(t).iloc[0]),
                         "mean_predicted": pred_curve[k + 1]})
    ax.set_xlim(0, t_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Years from diagnosis", fontsize=11)
    ax.set_ylabel("Cumulative rupture risk (%)", fontsize=11)
    title(ax, None, "Observed vs predicted cumulative rupture risk by risk group")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
    style(ax)
    save(fig, path)
    return pd.DataFrame(rows)