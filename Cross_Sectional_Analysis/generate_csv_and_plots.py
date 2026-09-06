"""
Figures and tables for the bAVM cross-sectional analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import roc_curve
HOPKINS_RED = '#9D2235'
INK = '#1A1A1A'
OTHER_COLORS = ['#4C72B0', '#55A868', '#C28E0E', '#8172B3', '#17A2B8', '#937860']
SHAP_CMAP = LinearSegmentedColormap.from_list('shap',
                                              ['#2E6FB7', '#C8CBD2', '#B5253A'])
DISPLAY_NAMES = {
    'Size': 'Nidus size (cm)',
    'Age': 'Age at diagnosis',
    'Deep_Location': 'Deep location',
    'Race_nonwhite': 'Non-white race',
    'Exclusive_Deep_Drainage': 'Exclusive deep drainage',
    'Single_Feeding_Artery': 'Single feeding artery',
    'Intranidal_Aneurysm': 'Intranidal aneurysm',
    'HTN': 'Hypertension',
    'Feeder_Count_Unknown': 'Feeder count undocumented',
}

def _label(name):
    return DISPLAY_NAMES.get(name, name.replace('_', ' '))

def _is_binary_col(col):
    return len(np.unique(col[~np.isnan(col)])) <= 2

def _beeswarm_y(xvals, row, max_spread=0.36, nbins=50):
    y_ = np.full(len(xvals), float(row))
    if len(xvals) < 2 or xvals.max() == xvals.min():
        return y_
    edges = np.linspace(xvals.min(), xvals.max(), nbins + 1)
    idx = np.clip(np.digitize(xvals, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    kmax = max(counts.max(), 1)
    for b in np.unique(idx):
        pos = np.where(idx == b)[0]
        k = len(pos)
        if k < 2:
            continue
        w = max_spread * np.sqrt(k / kmax)
        offs = np.linspace(-w, w, k)
        y_[pos] = row + offs[np.random.default_rng(b).permutation(k)]
    return y_

def _draw_beeswarm(ax, sv, Xvals, feature_names, raw_values=None,
                   max_points=600, show_importance=True):
    """SHAP beeswarm ordered by mean |SHAP|"""
    vals = raw_values if raw_values is not None else Xvals
    n = sv.shape[0]
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        sv, vals = sv[sel], vals[sel]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)
    for row, fi in enumerate(order):
        xv = sv[:, fi]
        yv = _beeswarm_y(xv, row)
        fv = np.asarray(vals[:, fi], dtype=float)
        if _is_binary_col(fv):
            lo_m, hi_m = fv == fv.min(), fv == fv.max()
            ax.scatter(xv[lo_m], yv[lo_m], color='#2E6FB7', s=13, alpha=0.62,
                       linewidths=0, rasterized=True, zorder=2)
            ax.scatter(xv[hi_m], yv[hi_m], color='#B5253A', s=13, alpha=0.62,
                       linewidths=0, rasterized=True, zorder=2)
        else:
            lo, hi = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
            ax.scatter(xv, yv, c=fv, cmap=SHAP_CMAP,
                       norm=Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1),
                       s=13, alpha=0.78, linewidths=0, rasterized=True, zorder=2)
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color='#000000', alpha=0.028, zorder=0)
    ax.axvline(0, color='#5A5A5A', lw=1.1, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(feature_names[i]) for i in order], fontsize=10.5)
    ax.set_ylim(-0.62, len(order) - 0.38)
    if show_importance:
        xmax = ax.get_xlim()[1]
        pad = (xmax - ax.get_xlim()[0]) * 0.02
        for row, fi in enumerate(order):
            ax.text(xmax + pad, row, f"{imp[fi]:.3f}", va='center', ha='left',
                    fontsize=8.5, color='#6B6B6B', family='monospace')
        ax.text(xmax + pad, len(order) - 0.15, "mean |SHAP|", va='bottom',
                ha='left', fontsize=8, color='#6B6B6B', style='italic')
    ax.set_xlabel('SHAP value  (log-odds contribution to rupture)',
                  fontsize=11.5, labelpad=26)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#BFBFBF')
    ax.tick_params(left=False, labelsize=9.5, colors='#3A3A3A')
    ax.grid(True, axis='x', alpha=0.14, lw=0.8)
    ax.set_axisbelow(True)
    ax.annotate('', xy=(0.34, -0.085), xytext=(0.05, -0.085),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#2E6FB7', lw=1.3))
    ax.text(0.355, -0.085, 'lower risk', transform=ax.transAxes,
            fontsize=9.5, color='#2E6FB7', va='center', ha='left')
    ax.annotate('', xy=(0.95, -0.085), xytext=(0.66, -0.085),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#B5253A', lw=1.3))
    ax.text(0.645, -0.085, 'higher risk', transform=ax.transAxes,
            fontsize=9.5, color='#B5253A', va='center', ha='right')

    sm = ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.028, pad=0.10, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Low', 'High'], fontsize=9)
    cbar.set_label('Feature value', fontsize=9.5)
    cbar.outline.set_visible(False)
    cbar.ax.text(1.9, 0.5, 'binary: absent / present', transform=cbar.ax.transAxes,
                 rotation=90, va='center', ha='center', fontsize=7.5,
                 color='#8A8A8A', style='italic')

ROC_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
              '#9B59B6', '#1ABC9C', '#E67E22']


def make_roc_figure(y, oof, nested_auc, ranked, fname='roc_all_models_nested.png'):
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, name in enumerate(ranked):
        fpr, tpr, _ = roc_curve(y, oof[name])
        ax.plot(fpr, tpr, color=ROC_COLORS[i % len(ROC_COLORS)], lw=3,
                label=f'{name} (AUC={nested_auc[name]:.3f})', alpha=0.9)
    fpr_r, tpr_r, _ = roc_curve(y, oof['R2eD'])
    ax.plot(fpr_r, tpr_r, 'k--', lw=3.2,
            label=f"R2eD (AUC={nested_auc['R2eD']:.3f})")
    ax.plot([0, 1], [0, 1], 'k:', lw=1.5, alpha=0.5, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'ROC - nested CV, all {len(y)} patients out-of-fold',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def make_shap_beeswarm(sv, Xraw, feature_names, model_name, fname):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    _draw_beeswarm(ax, sv, Xraw, feature_names, raw_values=Xraw)
    ax.set_title(f'Feature contributions - {model_name} (SHAP)',
                 fontsize=12.5, fontweight='bold', loc='left')
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fname

def make_risk_band_figure(y, p, bands, band_order, low_cut, high_cut,
                          model_name, fname):
    """Distribution of predicted risk and predicted vs observed per band."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Per-patient rupture probability - {model_name} (out-of-fold)",
                 fontsize=14, fontweight='bold')
    for cat, c in [('Low Risk', '#4CAF50'), ('Medium Risk', '#FF9800'),
                   ('High Risk', '#F44336')]:
        v = p[bands == cat]
        if len(v):
            axes[0].hist(v, bins=25, alpha=0.7, color=c, label=cat,
                         edgecolor='white')
    axes[0].axvline(low_cut, color='black', ls='--', alpha=0.7, lw=1.5)
    axes[0].axvline(high_cut, color='black', ls='--', alpha=0.7, lw=1.5)
    axes[0].set(xlabel='Predicted probability', ylabel='# patients')
    axes[0].set_title('Distribution of predicted risk')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    pr = [100 * p[bands == c].mean() if (bands == c).sum() else 0
          for c in band_order]
    ar = [100 * y[bands == c].mean() if (bands == c).sum() else 0
          for c in band_order]
    xp = np.arange(len(band_order))
    axes[1].bar(xp - 0.175, pr, 0.35, label='Mean predicted',
                color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.7)
    axes[1].bar(xp + 0.175, ar, 0.35, label='Observed',
                color=['#388E3C', '#F57C00', '#D32F2F'], alpha=0.9)
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(band_order)
    axes[1].set(xlabel='Risk band', ylabel='Rate (%)')
    axes[1].set_title('Predicted vs observed')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def make_sens_spec_figure(clinical_df, operating_points,
                          fname='sensitivity_specificity_by_model.png'):
    """Sensitivity and specificity per model at the sensitivity-targeted
    cutoffs"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, op in zip(axes, operating_points):
        sub = clinical_df[clinical_df['Operating_Point'] == op].copy()
        sub = sub.sort_values('Sensitivity', ascending=False).reset_index(drop=True)
        xpos = np.arange(len(sub))
        b1 = ax.bar(xpos - 0.175, sub['Sensitivity'], 0.35, label='Sensitivity',
                    color='#E74C3C', alpha=0.85)
        b2 = ax.bar(xpos + 0.175, sub['Specificity'], 0.35, label='Specificity',
                    color='#3498DB', alpha=0.85)
        for bars, vals in [(b1, sub['Sensitivity']), (b2, sub['Specificity'])]:
            for b, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.0f}',
                            ha='center', fontsize=8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(sub['Model'], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('% (pooled across outer folds)')
        ax.set_title(f'{op}\n(threshold locked per fold on inner OOF)',
                     fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
    fig.suptitle('Sensitivity / specificity at target operating points - nested CV',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def make_fold_stability_figure(fold_auc, names, n_outer,
                               fname='fold_stability_boxplot.png'):
    """Per-fold AUC spread"""
    order = sorted(names, key=lambda n: np.mean(fold_auc[n]))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    data = [np.array(fold_auc[n]) for n in order]
    try:
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55,
                        medianprops=dict(color=HOPKINS_RED, linewidth=2.2))
    except TypeError:
        bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.55,
                        medianprops=dict(color=HOPKINS_RED, linewidth=2.2))
    for patch, n in zip(bp['boxes'], order):
        patch.set_facecolor('#1F2937' if n == 'R2eD' else HOPKINS_RED)
        patch.set_alpha(0.45)
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Outer-fold AUC spread ({n_outer} folds)\n'
                 'the height of each box is how much a single split '
                 'could have moved things',
                 fontsize=12.5, fontweight='bold', loc='left')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def make_delong_heatmap(matrix, label, fname):
    """Pairwise DeLong p or BH-adjusted q values"""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.2,
                cbar_kws={'label': label}, linewidths=0.5, ax=ax)
    ax.set_title(f'Pairwise DeLong - {label} (nested CV)', fontsize=12,
                 fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname
