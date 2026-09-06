"""
RISK BANDS FOR EVERY MODEL
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

# Band cutoffs
LOW_CUT, HIGH_CUT = 0.20, 0.50
BANDS = ['Low Risk', 'Medium Risk', 'High Risk']
HOPKINS_RED = '#9D2235'
BAND_COLORS = {'Low Risk': '#4C9A5E', 'Medium Risk': '#E0A030',
               'High Risk': '#B5253A'}

def assign_bands(p):
    return np.where(p < LOW_CUT, 'Low Risk',
                    np.where(p < HIGH_CUT, 'Medium Risk', 'High Risk'))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = Path(sys.argv[1])
    src = folder / 'nested_oof_predictions.csv'
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    df = pd.read_csv(src)
    y = df['y_true'].to_numpy(dtype=int)
    models = [c for c in df.columns if c not in ('y_true', 'Fold')]
    preds = {m: df[m].to_numpy(dtype=float) for m in models}
    models = sorted(models, key=lambda m: roc_auc_score(y, preds[m]), reverse=True)
    print(f"RISK BANDS  |  {len(y)} patients, {int(y.sum())} ruptured "
          f"({100 * y.mean():.1f}%)")
    print(f"Cutoffs: <{LOW_CUT} low, {LOW_CUT}-{HIGH_CUT} medium, "
          f">={HIGH_CUT} high")
    rows, summary = [], []
    for m in models:
        p = preds[m]
        b = assign_bands(p)
        auc = roc_auc_score(y, p)
        weighted_gap, total = 0.0, 0
        print(f"\n{m}  (AUC {auc:.3f}, Brier {brier_score_loss(y, p):.3f})")
        for band in BANDS:
            mask = b == band
            n = int(mask.sum())
            if n == 0:
                print(f"  {band:>12}: empty")
                rows.append({'Model': m, 'Band': band, 'N': 0,
                             'Pct_of_cohort': 0.0, 'Mean_predicted': np.nan,
                             'Observed': np.nan, 'Gap': np.nan})
                continue
            pred = 100 * p[mask].mean()
            obs = 100 * y[mask].mean()
            gap = abs(pred - obs)
            weighted_gap += gap * n
            total += n
            print(f"  {band:>12}: n={n:>4} ({100 * n / len(y):>4.1f}%) | "
                  f"predicted {pred:>5.1f}% | observed {obs:>5.1f}% | "
                  f"gap {pred - obs:>+5.1f}")
            rows.append({'Model': m, 'Band': band, 'N': n,
                         'Pct_of_cohort': round(100 * n / len(y), 1),
                         'Mean_predicted': round(pred, 1),
                         'Observed': round(obs, 1),
                         'Gap': round(pred - obs, 1)})

        # Spread between the low and high band observed rates: how much
        # clinical separation the bands actually deliver
        lo_obs = 100 * y[b == 'Low Risk'].mean() if (b == 'Low Risk').any() else np.nan
        hi_obs = 100 * y[b == 'High Risk'].mean() if (b == 'High Risk').any() else np.nan
        summary.append({
            'Model': m,
            'AUC': round(auc, 4),
            'Brier': round(brier_score_loss(y, p), 4),
            'Mean_abs_gap': round(weighted_gap / max(total, 1), 2),
            'Low_band_observed': round(lo_obs, 1) if not np.isnan(lo_obs) else np.nan,
            'High_band_observed': round(hi_obs, 1) if not np.isnan(hi_obs) else np.nan,
            'Band_separation': (round(hi_obs - lo_obs, 1)
                                if not (np.isnan(hi_obs) or np.isnan(lo_obs))
                                else np.nan),
            'N_bands_used': int(sum((b == x).any() for x in BANDS)),
        })

    band_df = pd.DataFrame(rows)
    band_df.to_csv(folder / 'risk_bands_all_models.csv', index=False)
    sum_df = pd.DataFrame(summary).sort_values('Mean_abs_gap')
    sum_df.to_csv(folder / 'risk_bands_summary.csv', index=False)
    print("SUMMARY")
    print(sum_df.to_string(index=False))

    # figure 
    n_m = len(models)
    ncol = 4
    nrow = int(np.ceil(n_m / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // ncol, i % ncol]
        sub = band_df[band_df['Model'] == m].set_index('Band').reindex(BANDS)
        xp = np.arange(3)
        ax.bar(xp - 0.19, sub['Mean_predicted'], 0.38, label='Predicted',
               color=[BAND_COLORS[b] for b in BANDS], alpha=0.55)
        ax.bar(xp + 0.19, sub['Observed'], 0.38, label='Observed',
               color=[BAND_COLORS[b] for b in BANDS], alpha=0.95)
        for x, (pr, ob, n) in enumerate(zip(sub['Mean_predicted'],
                                            sub['Observed'], sub['N'])):
            if not np.isnan(ob):
                ax.text(x, max(pr, ob) + 3, f"n={int(n)}", ha='center',
                        fontsize=7.5, color='#555555')
        ax.set_xticks(xp)
        ax.set_xticklabels(['Low', 'Med', 'High'], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Rupture rate (%)', fontsize=9)
        gap = sum_df.set_index('Model').loc[m, 'Mean_abs_gap']
        ax.set_title(f"{m}\nmean gap {gap:.1f} pts", fontsize=10,
                     fontweight='bold' if m == models[0] else 'normal')
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    for j in range(n_m, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')

    fig.suptitle('Predicted vs observed rupture rate by risk band '
                 '(nested CV, out-of-fold)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(folder / 'risk_bands_all_models.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
