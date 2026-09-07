"""
Figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

TIER_COLORS = {"Low": "#3E8E5A", "Medium": "#D9922B", "High": "#B5253A"}
TIER_ORDER = ["Low", "Medium", "High"]
NAVY = "#0E2A47"
CRIMSON = "#9D2235"
MUTED = "#7f8c8d"
def _panel_km(ax, df, tier, logrank_p, cindex, ci_lo, ci_hi, t_max=10.0,
              fu_mean=None, fu_sd=None, fu_median=None, fu_iqr=None):
    for t in TIER_ORDER:
        m = tier == t
        if m.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[m, "time"], df.loc[m, "event"],
                label=f"{t} (n={int(m.sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=TIER_COLORS[t],
                                   linewidth=2.0, ci_alpha=0.10)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Years from diagnosis", fontsize=11)
    ax.set_ylabel("Cumulative rupture-free survival", fontsize=11)
    p_txt = "< 0.001" if logrank_p < 0.001 else f"= {logrank_p:.3f}"
    txt = (f"Log-rank p {p_txt}\n"
           f"C = {cindex:.3f} (95% CI {ci_lo:.3f}-{ci_hi:.3f})\n"
           f"N = {len(df)}, events = {int(df['event'].sum())}")
    if fu_mean is not None:
        txt += f"\nMean FU = {fu_mean:.1f}y (SD {fu_sd:.1f})"
    if fu_median is not None:
        txt += f"\nMedian FU = {fu_median:.1f}y (IQR {fu_iqr})"
    ax.text(0.03, 0.24, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", bbox=dict(facecolor="white", alpha=0.9,
                                   edgecolor="#BBBBBB", boxstyle="square,pad=0.5"))
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.03, 0.03), fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor="#BBBBBB",
                    borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(0.8)
    ax.grid(ls=":", alpha=0.3)

def _panel_rate(ax, rate_df):
    xp = np.arange(len(rate_df))
    ax.bar(xp, rate_df["rate_pct"], 0.55,
           color=[TIER_COLORS[t] for t in rate_df["Tier"]], alpha=0.85)
    ax.errorbar(xp, rate_df["rate_pct"],
                yerr=[rate_df["rate_pct"] - rate_df["lo_pct"],
                      rate_df["hi_pct"] - rate_df["rate_pct"]],
                fmt="none", ecolor="#3A3A3A", capsize=4, lw=1.2)
    for x, r in zip(xp, rate_df.itertuples()):
        ax.text(x, r.hi_pct + 0.4,
                f"{r.rate_pct:.1f}%\n(n={r.N}, ev={r.Events})",
                ha="center", fontsize=8.5)
    ax.set_xticks(xp)
    ax.set_xticklabels([f"{t}\n({lbl})" for t, lbl in
                        zip(rate_df["Tier"], rate_df["Score_range"])],
                       fontsize=10)
    ax.set_ylabel("Annual hemorrhage rate (%)", fontsize=11)
    ax.set_xlabel("Risk tier", fontsize=11)
    ax.set_ylim(0, max(rate_df["hi_pct"]) * 1.35)
    ax.grid(ls=":", alpha=0.3, axis="y")

def _panel_calibration(ax, calib_df, slope, intercept, horizon, cap=None,
                       tick_step=0.02):
    """Predicted vs observed at one horizon.
    """
    if cap is None:
        cap = max(calib_df["pred"].max(), calib_df["obs"].max()) * 1.20
        cap = float(np.ceil(cap / tick_step) * tick_step)
    ax.plot([0, cap], [0, cap], "--", color=MUTED, lw=1.3, zorder=1)
    ax.plot(calib_df["pred"], calib_df["obs"], "-", color="#7FB3D8", lw=1.6,
            zorder=2)
    half = cap * 0.014
    for xv, ov, lov in zip(calib_df["pred"], calib_df["obs"],
                           calib_df["obs_lo"]):
        ax.vlines(xv, lov, ov, color=NAVY, lw=1.5, zorder=3)
        ax.hlines(lov, xv - half, xv + half, color=NAVY, lw=1.5, zorder=3)
    ax.plot(calib_df["pred"], calib_df["obs"], "o", color=NAVY, markersize=10,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ticks = np.arange(0, cap + 1e-9, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t * 100:.0f}" for t in ticks])
    ax.set_yticklabels([f"{t * 100:.0f}" for t in ticks])
    ax.set_xlim(0, cap)
    ax.set_ylim(0, cap)
    ax.set_xlabel(f"Predicted {horizon:.0f}-year rupture risk (%)", fontsize=11)
    ax.set_ylabel(f"Observed {horizon:.0f}-year rupture rate (%)", fontsize=11)
    ax.text(0.04, 0.93, f"Slope = {slope:.2f}\nIntercept = {intercept:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))
    ax.grid(ls=":", alpha=0.3)

def _panel_tdauc(ax, times, auc_model, auc_lo, auc_hi, auc_r2ed=None,
                 r2ed_lo=None, r2ed_hi=None):
    ax.plot(times, auc_model, "o-", color=NAVY, lw=2, markersize=5,
            label="3-feature Ridge Cox")
    if auc_lo is not None:
        ax.fill_between(times, auc_lo, auc_hi, color=NAVY, alpha=0.15)
    if auc_r2ed is not None:
        ax.plot(times, auc_r2ed, "s-", color="#C28E0E", lw=1.6, markersize=4,
                label="R2eD score")
        if r2ed_lo is not None:
            ax.fill_between(times, r2ed_lo, r2ed_hi, color="#C28E0E",
                            alpha=0.12)
    ax.axhline(0.5, ls=":", color="#5B9BD5", lw=1)
    ax.set_xlabel("Time horizon (years)", fontsize=11)
    ax.set_ylabel("Time-dependent AUC", fontsize=11)
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(ls=":", alpha=0.3)

def figure_longitudinal(df, tier, logrank_p, cindex, ci_lo, ci_hi,
                        rate_df, calib_df, slope, intercept, horizon,
                        times, auc_model, auc_lo, auc_hi, auc_r2ed,
                        r2ed_lo=None, r2ed_hi=None,
                        fname="FIGURE_longitudinal.png", calib_cap=None,
                        fu_mean=None, fu_sd=None, fu_median=None,
                        fu_iqr=None):
    """Four panels: KM by tier, annual rate, calibration, time-dependent AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.26})
    (axA, axB), (axC, axD) = axes

    _panel_km(axA, df, tier, logrank_p, cindex, ci_lo, ci_hi,
              fu_mean=fu_mean, fu_sd=fu_sd, fu_median=fu_median,
              fu_iqr=fu_iqr)
    _panel_rate(axB, rate_df)
    _panel_calibration(axC, calib_df, slope, intercept, horizon, calib_cap)
    _panel_tdauc(axD, times, auc_model, auc_lo, auc_hi, auc_r2ed,
                 r2ed_lo, r2ed_hi)

    for ax, letter, title in [
            (axA, "A", "Kaplan-Meier by risk score"),
            (axB, "B", "Annual rate by risk tier"),
            (axC, "C", f"Calibration at {horizon:.0f} years"),
            (axD, "D", "Time-dependent AUC")]:
        ax.set_title(f"{letter}. {title}", fontsize=12.5, fontweight="bold",
                     loc="left", pad=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fname

def figure_km_untreated(df, tier, fname="km_untreated.png"):
    """Same stratification applied to the untreated subset."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for t in TIER_ORDER:
        m = tier == t
        if m.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[m, "time"], df.loc[m, "event"],
                label=f"{t} (n={int(m.sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=TIER_COLORS[t],
                                   linewidth=2.0)
    ax.set_xlabel("Years from diagnosis", fontsize=11)
    ax.set_ylabel("Cumulative rupture-free survival", fontsize=11)
    ax.set_title("Untreated subset, model applied without re-estimation",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname

def figure_score_distribution(score, rate_by_score,
                              fname="score_distribution.png"):
    """How the integer score is distributed, and the rate at each level.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    vals = np.arange(int(np.nanmin(score)), int(np.nanmax(score)) + 1)
    counts = [int(np.sum(score == v)) for v in vals]
    axL.bar(vals, counts, 0.62, color=NAVY, alpha=0.8)
    for v, c in zip(vals, counts):
        axL.text(v, c + max(counts) * 0.015,
                 f"{c}\n({100 * c / len(score):.1f}%)",
                 ha="center", fontsize=8.5, color=MUTED)
    axL.set_xticks(vals)
    axL.set_xlabel("Integer score", fontsize=11)
    axL.set_ylabel("Patients", fontsize=11)
    axL.set_ylim(0, max(counts) * 1.18)
    axL.set_title("A. Score distribution", fontsize=12.5, fontweight="bold",
                  loc="left")
    axL.grid(ls=":", alpha=0.3, axis="y")
    axR.bar(rate_by_score["Score"], rate_by_score["rate_pct"], 0.62,
            color=CRIMSON, alpha=0.8)
    axR.errorbar(rate_by_score["Score"], rate_by_score["rate_pct"],
                 yerr=[rate_by_score["rate_pct"] - rate_by_score["lo_pct"],
                       rate_by_score["hi_pct"] - rate_by_score["rate_pct"]],
                 fmt="none", ecolor="#3A3A3A", elinewidth=1.2, capsize=0)
    for r in rate_by_score.itertuples():
        axR.text(r.Score, r.hi_pct + 0.3, f"n={r.N}, ev={r.Events}",
                 ha="center", fontsize=8, color=MUTED)
    axR.set_xticks(rate_by_score["Score"])
    axR.set_xlabel("Integer score", fontsize=11)
    axR.set_ylabel("Annual hemorrhage rate (%)", fontsize=11)
    axR.set_title("B. Annual rate by score", fontsize=12.5, fontweight="bold",
                  loc="left")
    axR.grid(ls=":", alpha=0.3, axis="y")
    for ax in (axL, axR):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fname