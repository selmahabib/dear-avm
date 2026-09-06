"""
Rank stability: how often is each model on top?
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
RANDOM_STATE = 42
HOPKINS_RED = '#9D2235'
def stratified_boot_idx(y, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = Path(sys.argv[1])
    oof_path = folder / 'nested_oof_predictions.csv'
    if not oof_path.exists():
        print(f"Not found: {oof_path}")
        return 1
    df = pd.read_csv(oof_path)
    y = df['y_true'].to_numpy(dtype=int)
    models = [c for c in df.columns if c not in ('y_true', 'Fold')]
    preds = {m: df[m].to_numpy(dtype=float) for m in models}
    print(f"RANK STABILITY  |  {len(y)} patients, {int(y.sum())} ruptured, "
          f"{len(models)} models")
    point = {m: roc_auc_score(y, preds[m]) for m in models}
    ranked = sorted(models, key=lambda m: point[m], reverse=True)
    print("\nPoint estimates:")
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m:<22} {point[m]:.4f}")

    # patient-level bootstrap 
    rng = np.random.default_rng(RANDOM_STATE)
    top1 = {m: 0 for m in models}
    top3 = {m: 0 for m in models}
    rank_sum = {m: 0 for m in models}
    beats = {(a, b): 0 for a in models for b in models if a != b}
    n_ok = 0
    for _ in range(N_BOOT):
        idx = stratified_boot_idx(y, rng)
        yb = y[idx]
        try:
            aucs = {m: roc_auc_score(yb, preds[m][idx]) for m in models}
        except ValueError:
            continue
        n_ok += 1
        order = sorted(models, key=lambda m: aucs[m], reverse=True)
        top1[order[0]] += 1
        for m in order[:3]:
            top3[m] += 1
        for r, m in enumerate(order, 1):
            rank_sum[m] += r
        for a, b in combinations(models, 2):
            if aucs[a] > aucs[b]:
                beats[(a, b)] += 1
            else:
                beats[(b, a)] += 1

    rows = [{'Model': m,
             'AUC': round(point[m], 4),
             'Top1_%': round(100 * top1[m] / n_ok, 1),
             'Top3_%': round(100 * top3[m] / n_ok, 1),
             'Mean_rank': round(rank_sum[m] / n_ok, 2)}
            for m in models]
    rs = pd.DataFrame(rows).sort_values('Top1_%', ascending=False).reset_index(drop=True)
    rs.to_csv(folder / 'rank_stability_bootstrap.csv', index=False)

    print(f"\n--- Patient-level bootstrap ({n_ok} resamples) ---")
    print(rs.to_string(index=False))

    #  pairwise win probability 
    W = pd.DataFrame(np.full((len(models), len(models)), np.nan),
                     index=models, columns=models)
    for a, b in combinations(models, 2):
        W.loc[a, b] = round(100 * beats[(a, b)] / n_ok, 1)
        W.loc[b, a] = round(100 * beats[(b, a)] / n_ok, 1)
    W = W.loc[ranked, ranked]
    W.to_csv(folder / 'rank_stability_pairwise_wins.csv')
    print("\n--- P(row model has higher AUC than column model), % ---")
    print(W.to_string())

    # fold-level ranking 
    fold_path = folder / 'nested_cv_per_fold.csv'
    if fold_path.exists():
        fd = pd.read_csv(fold_path)
        piv = fd.pivot_table(index='Fold', columns='Model', values='Outer_test_AUC')
        piv = piv[[m for m in models if m in piv.columns]]
        fold_ranks = piv.rank(axis=1, ascending=False)
        frows = [{'Model': m,
                  'Fold_wins': int((fold_ranks[m] == 1).sum()),
                  'N_folds': int(fold_ranks[m].notna().sum()),
                  'Mean_fold_rank': round(float(fold_ranks[m].mean()), 2),
                  'Best_fold_AUC': round(float(piv[m].max()), 3),
                  'Worst_fold_AUC': round(float(piv[m].min()), 3)}
                 for m in piv.columns]
        fr = pd.DataFrame(frows).sort_values(
            ['Fold_wins', 'Mean_fold_rank'], ascending=[False, True])
        fr.to_csv(folder / 'rank_stability_by_fold.csv', index=False)
        print("\n--- Fold-level: how many outer folds each model won ---")
        print(fr.to_string(index=False))

    # figure 
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={'width_ratios': [1, 1.15]})

    plot_order = rs.sort_values('Top1_%')['Model'].tolist()
    colors = [HOPKINS_RED if m == ranked[0] else '#B8B8B8' for m in plot_order]
    axL.barh(range(len(plot_order)), [rs.set_index('Model').loc[m, 'Top1_%']
                                      for m in plot_order],
             color=colors, alpha=0.85)
    axL.set_yticks(range(len(plot_order)))
    axL.set_yticklabels(plot_order, fontsize=10)
    axL.set_xlabel('% of bootstrap resamples ranked first', fontsize=11)
    axL.set_title('A. How often each model is best', fontsize=12,
                  fontweight='bold', loc='left')
    axL.axvline(100 / len(models), color='#666666', ls=':', lw=1.2)
    axL.text(100 / len(models), len(plot_order) - 0.4, ' chance',
             fontsize=8.5, color='#666666', va='top')
    for sp in ('top', 'right'):
        axL.spines[sp].set_visible(False)
    axL.grid(True, axis='x', alpha=0.2)
    axL.set_axisbelow(True)

    rng2 = np.random.default_rng(RANDOM_STATE + 1)
    dists = {m: [] for m in models}
    for _ in range(600):
        idx = stratified_boot_idx(y, rng2)
        for m in models:
            try:
                dists[m].append(roc_auc_score(y[idx], preds[m][idx]))
            except ValueError:
                pass
    box_order = sorted(models, key=lambda m: point[m])
    bp = axR.boxplot([dists[m] for m in box_order], vert=False,
                     patch_artist=True, widths=0.6,
                     medianprops=dict(color='black', lw=1.4))
    for patch, m in zip(bp['boxes'], box_order):
        patch.set_facecolor(HOPKINS_RED if m == ranked[0] else '#B8B8B8')
        patch.set_alpha(0.6)
    axR.set_yticklabels(box_order, fontsize=10)
    axR.set_xlabel('Bootstrap AUC', fontsize=11)
    axR.set_title('B. Overlap between models', fontsize=12,
                  fontweight='bold', loc='left')
    for sp in ('top', 'right'):
        axR.spines[sp].set_visible(False)
    axR.grid(True, axis='x', alpha=0.2)
    axR.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(folder / 'rank_stability.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())