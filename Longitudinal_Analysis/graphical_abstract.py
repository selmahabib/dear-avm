"""Draws the graphical-abstract KM plot (Figure 3C curves, no CI bands or risk table, transparent background).

Run after main.py: python graphical_abstract.py <results folder> [--data registry.xlsx]

Writes to <results folder>/graphical_abstract/:
  GA_km_tiers.svg / .pdf / .png   transparent, ready for PowerPoint
  GA_km_tiers_curves.csv          plotted step data, to check against Figure 3C
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import config as cfg
from bavm.data import load_and_engineer
from bavm.episodes import split
from bavm.model import score_with
from bavm.plotting import TIER_COLORS, TIER_ORDER
from bavm.stats import score_tier

# Styling (curves do not change)
T_MAX = 10.0
Y_MIN = 0.5
LINE_WIDTH = 2.6
FONT = "Arial"
FONT_SIZE = 11
INK = "#1A1A1A"
SHOW_LOGRANK = True
LOGRANK_TEXT = "Log-rank P < 0.001"   # from the manuscript / main.py output
FIG_W, FIG_H = 4.6, 3.6


def main(results, data_path):
    out_dir = os.path.join(results, "graphical_abstract")
    os.makedirs(out_dir, exist_ok=True)

    # Score points from the main run
    pts_path = os.path.join(results, "tv_model_score_points.csv")
    if not os.path.exists(pts_path):
        raise SystemExit(f"Missing {pts_path}; run main.py first.")
    row = pd.read_csv(pts_path).query("Score == 'Time-varying'").iloc[0]
    pd_tv, pe_tv, pl_tv = (int(row["Deep_location"]), int(row["Prior_rupture_early"]),
                           int(row["Prior_rupture_late"]))
    cut = float(row["cut_yr"]) if np.isfinite(row["cut_yr"]) else cfg.FIXED_CUT
    print(f"Score points: deep={pd_tv}, prior early={pe_tv}, prior late={pl_tv}, cut={cut:g} yr")

    # Data
    df = load_and_engineer(data_path)
    keep = df[["time", "event"] + cfg.BASE].notna().all(axis=1).to_numpy()
    d = df.loc[keep, ["time", "event"] + cfg.BASE].astype(float).reset_index(drop=True)
    d["time"] = d["time"].clip(lower=cfg.MIN_TIME)
    s_e, s_l = score_with(d, pd_tv, pe_tv), score_with(d, pd_tv, pl_tv)
    print(f"Cohort: n={len(d)}, events={int(d['event'].sum())}")

    # Time-updated tiers, as in Figure 3C
    ep = split(d.assign(_se=s_e, _sl=s_l), ["_se", "_sl"], [cut])
    ep["score"] = np.where(ep["interval"] == 0, ep["_se"], ep["_sl"])
    ep["tier"] = np.asarray(score_tier(ep["score"].to_numpy()))

    # Plot
    plt.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    curves, labels = [], []
    for t in TIER_ORDER:
        m = ep["tier"] == t
        if not m.any():
            continue
        lo_s, hi_s = int(ep.loc[m, "score"].min()), int(ep.loc[m, "score"].max())
        rng = f"{lo_s}" if lo_s == hi_s else f"{lo_s}–{hi_s}"
        kmf = KaplanMeierFitter().fit(ep.loc[m, "stop"], ep.loc[m, "_ev"],
                                      entry=ep.loc[m, "start"], label=t)
        sf = kmf.survival_function_.iloc[:, 0]
        ax.step(sf.index, sf.values, where="post", color=TIER_COLORS[t], lw=LINE_WIDTH,
                solid_capstyle="round", solid_joinstyle="round")
        curves.append(pd.DataFrame({"tier": t, "time": sf.index, "survival": sf.values}))
        labels.append((t, f"{t} ({rng})", float(sf[sf.index <= T_MAX].iloc[-1])))

    # Direct labels at the right end, nudged apart if they overlap
    labels.sort(key=lambda r: r[2])
    min_gap = 0.045 * (1 - Y_MIN)
    ys = [r[2] for r in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (t, lab, _), y in zip(labels, ys):
        ax.text(T_MAX + 0.15, y, lab, color=TIER_COLORS[t], va="center", ha="left",
                fontsize=FONT_SIZE, fontweight="bold", clip_on=False)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(Y_MIN, 1.0)
    ax.set_xticks(np.arange(0, T_MAX + 1e-9, 2))
    ax.set_yticks(np.arange(Y_MIN, 1.001, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(Y_MIN, 1.001, 0.1)])
    ax.set_xlabel("Years from diagnosis", color=INK)
    ax.set_ylabel("Rupture-free survival", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK, length=3, width=0.8)
    ax.grid(False)
    if SHOW_LOGRANK:
        ax.text(0.03, 0.04, LOGRANK_TEXT, transform=ax.transAxes, color=INK,
                fontsize=FONT_SIZE - 1, ha="left", va="bottom")

    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"GA_km_tiers.{ext}"), dpi=300,
                    transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    pd.concat(curves).to_csv(os.path.join(out_dir, "GA_km_tiers_curves.csv"), index=False)
    print(f"Saved GA_km_tiers.svg/.pdf/.png -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results folder written by main.py")
    ap.add_argument("--data", default=cfg.DATA_PATH)
    args = ap.parse_args()
    main(os.path.abspath(args.results), args.data)