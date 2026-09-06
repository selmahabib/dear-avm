"""
MAIN PUBLICATION FIGURE 
"""
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATHS = [
 r'path/to/your/data.xlsx'
]

LOW_CUT, HIGH_CUT = 0.20, 0.50
CLASS_WEIGHTING = False
RANDOM_STATE = 42
HOPKINS_RED = '#9D2235'
INK = '#1A1A1A'
MUTED = '#6B6B6B'
OTHER_COLORS = ['#4C72B0', '#55A868', '#C28E0E', '#8172B3', '#17A2B8', '#937860']
SHAP_CMAP = LinearSegmentedColormap.from_list('shap',
                                              ['#2E6FB7', '#C8CBD2', '#B5253A'])
BANDS = ['Low', 'Medium', 'High']
BAND_COLORS = {'Low': '#3E8E5A', 'Medium': '#D9922B', 'High': '#B5253A'}
DISPLAY_NAMES = {
    'Size': 'Nidus size (cm)',
    'Age': 'Age at\ndiagnosis',
    'Deep_Location': 'Deep location',
    'Race_nonwhite': 'Non-white race',
    'Exclusive_Deep_Drainage': 'Exclusive deep\ndrainage',
    'Single_Feeding_Artery': 'Single feeding\nartery',
    'Intranidal_Aneurysm': 'Intranidal\naneurysm',
    'HTN': 'Hypertension',
    'Feeder_Count_Unknown': 'Feeder count\nundocumented',
}
BASE_FEATURES = ['Race_nonwhite', 'Deep_Location', 'Size',
                 'Exclusive_Deep_Drainage', 'Single_Feeding_Artery',
                 'Feeder_Count_Unknown', 'Age', 'Intranidal_Aneurysm', 'HTN']

def _label(n):
    return DISPLAY_NAMES.get(n, n.replace('_', ' '))

def load_cohort():
    raw = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            raw = pd.read_excel(p)
            break
    if raw is None:
        return None, None
    d = raw.copy()
    d['Race_nonwhite'] = (d['Race'] != '1_White').astype(float)
    d.loc[d['Race'].isna(), 'Race_nonwhite'] = np.nan
    d['Age'] = pd.to_numeric(d['Age_dx'], errors='coerce')
    deep = [c for c in ['Location_brainstem', 'Location_basal_ganglia',
                        'Location_thalamus', 'Location_Cerebellar_hemisphere',
                        'Location_Cerebellar_vermis'] if c in d.columns]
    if deep:
        sub = d[deep]
        d['Deep_Location'] = sub.any(axis=1).astype(float)
        d.loc[sub.isna().all(axis=1), 'Deep_Location'] = np.nan
    d['Size'] = pd.to_numeric(d['max_Size'], errors='coerce')
    d['Size_less_3cm'] = (d['Size'] < 3).astype(float)
    d.loc[d['Size'].isna(), 'Size_less_3cm'] = np.nan
    if {'Drainage_vein_deep', 'Drainage_vein_superficial'} <= set(d.columns):
        dd = d[['Drainage_vein_deep', 'Drainage_vein_superficial']]
        d['Exclusive_Deep_Drainage'] = (
            (d['Drainage_vein_deep'] == 1) & (d['Drainage_vein_superficial'] == 0)
        ).astype(float)
        d.loc[dd.isna().any(axis=1), 'Exclusive_Deep_Drainage'] = np.nan
    for c in ['Feeder_Territory_Number', 'Feeder_Teriitory_Number']:
        if c in d.columns:
            ft = pd.to_numeric(d[c], errors='coerce')
            d['Single_Feeding_Artery'] = (ft == 1).astype(float)
            d['Feeder_Count_Unknown'] = ft.isna().astype(float)
            d.loc[ft.isna(), 'Single_Feeding_Artery'] = 0.0
            break
    if 'Aneurysm_intranidal' in d.columns:
        d['Intranidal_Aneurysm'] = pd.to_numeric(d['Aneurysm_intranidal'],
                                                 errors='coerce')
    if 'Hypertension' in d.columns:
        d['HTN'] = pd.to_numeric(d['Hypertension'], errors='coerce')
    d['Event'] = pd.to_numeric(d['Ever_Ruptured'], errors='coerce')
    feats = [f for f in BASE_FEATURES if f in d.columns]
    clean = d[feats + ['Size_less_3cm', 'Event']].dropna()
    return clean[feats].to_numpy(float), clean['Event'].to_numpy(int)


def build_model(name, params, seed=RANDOM_STATE):
    cw = 'balanced' if CLASS_WEIGHTING else None
    if name == 'CatBoost':
        import catboost as cb
        est = cb.CatBoostClassifier(iterations=300, random_state=seed,
                                    verbose=False, allow_writing_files=False,
                                    auto_class_weights='Balanced' if CLASS_WEIGHTING else None)
    elif name == 'Random Forest':
        est = RandomForestClassifier(n_estimators=300, random_state=seed,
                                     class_weight=cw, n_jobs=1)
    elif name == 'XGBoost':
        import xgboost as xgb
        est = xgb.XGBClassifier(n_estimators=300, random_state=seed,
                                eval_metric='logloss')
    elif name == 'LightGBM':
        import lightgbm as lgb
        est = lgb.LGBMClassifier(n_estimators=300, random_state=seed,
                                 verbose=-1, class_weight=cw)
    elif name == 'AdaBoost':
        est = AdaBoostClassifier(random_state=seed)
    elif name == 'SVM':
        est = SVC(kernel='rbf', probability=True, random_state=seed,
                  class_weight=cw)
    else:
        est = LogisticRegression(max_iter=2000, random_state=seed, class_weight=cw)
    return Pipeline([('scale', StandardScaler()), ('clf', est)]).set_params(**params)


# beeswarm helpers
def _beeswarm_y(xv, row, max_spread=0.36, nbins=50):
    y = np.full(len(xv), float(row))
    if len(xv) < 2 or xv.max() == xv.min():
        return y
    edges = np.linspace(xv.min(), xv.max(), nbins + 1)
    idx = np.clip(np.digitize(xv, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    kmax = max(counts.max(), 1)
    for b in np.unique(idx):
        pos = np.where(idx == b)[0]
        k = len(pos)
        if k < 2:
            continue
        w = max_spread * np.sqrt(k / kmax)
        offs = np.linspace(-w, w, k)
        y[pos] = row + offs[np.random.default_rng(b + row * 97).permutation(k)]
    return y


def draw_beeswarm(ax, sv, vals, names, max_points=600):
    n = sv.shape[0]
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        sv, vals = sv[sel], vals[sel]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)
    for row, fi in enumerate(order):
        xv, fv = sv[:, fi], np.asarray(vals[:, fi], float)
        yv = _beeswarm_y(xv, row)
        if len(np.unique(fv[~np.isnan(fv)])) <= 2:
            lo_m, hi_m = fv == fv.min(), fv == fv.max()
            ax.scatter(xv[lo_m], yv[lo_m], color='#2E6FB7', s=11, alpha=0.6,
                       linewidths=0, rasterized=True, zorder=2)
            ax.scatter(xv[hi_m], yv[hi_m], color='#B5253A', s=11, alpha=0.6,
                       linewidths=0, rasterized=True, zorder=2)
        else:
            lo, hi = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
            ax.scatter(xv, yv, c=fv, cmap=SHAP_CMAP,
                       norm=Normalize(lo, hi if hi > lo else lo + 1),
                       s=11, alpha=0.78, linewidths=0, rasterized=True, zorder=2)
        if row % 2 == 0:
            ax.axhspan(row - .5, row + .5, color='#000000', alpha=0.028, zorder=0)
    ax.axvline(0, color='#5A5A5A', lw=1.05, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(names[i]) for i in order], fontsize=9.5)
    ax.set_ylim(-0.62, len(order) - 0.38)
    xmax = ax.get_xlim()[1]
    pad = (xmax - ax.get_xlim()[0]) * 0.02
    for row, fi in enumerate(order):
        ax.text(xmax + pad, row, f"{imp[fi]:.3f}", va='center', ha='left',
                fontsize=8, color=MUTED, family='monospace')
    ax.set_xlabel('SHAP value  (log-odds contribution to rupture)', fontsize=10.5)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#BFBFBF')
    ax.tick_params(left=False, labelsize=9, colors='#3A3A3A')
    ax.grid(True, axis='x', alpha=0.14, lw=0.8)
    ax.set_axisbelow(True)
    sm = ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cb_ = ax.figure.colorbar(sm, ax=ax, fraction=0.026, pad=0.09, ticks=[0, 1])
    cb_.ax.set_yticklabels(['Low', 'High'], fontsize=8.5)
    cb_.set_label('Feature value', fontsize=9)
    cb_.outline.set_visible(False)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else 'CatBoost'

    oof = pd.read_csv(folder / 'nested_oof_predictions.csv')
    y = oof['y_true'].to_numpy(int)
    models = [c for c in oof.columns if c not in ('y_true', 'Fold')]
    preds = {m: oof[m].to_numpy(float) for m in models}
    aucs = {m: roc_auc_score(y, preds[m]) for m in models}
    ranked = sorted([m for m in models if m != 'R2eD'],
                    key=lambda m: aucs[m], reverse=True)

    p = preds[model]
    band = np.where(p < LOW_CUT, 'Low', np.where(p < HIGH_CUT, 'Medium', 'High'))
    stats = {b: {'n': int((band == b).sum()),
                 'pred': 100 * p[band == b].mean() if (band == b).any() else np.nan,
                 'obs': 100 * y[band == b].mean() if (band == b).any() else np.nan}
             for b in BANDS}

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.24,
                          height_ratios=[1.25, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A: ROC 
    handles = {}
    for i, m in enumerate(ranked):
        if m == model:
            continue
        fpr, tpr, _ = roc_curve(y, preds[m])
        (h,) = axA.plot(fpr, tpr, color=OTHER_COLORS[i % len(OTHER_COLORS)],
                        lw=1.6, alpha=0.75, zorder=1,
                        label=f'{m} ({aucs[m]:.3f})')
        handles[m] = h
    if 'R2eD' in preds:
        fpr, tpr, _ = roc_curve(y, preds['R2eD'])
        (h,) = axA.plot(fpr, tpr, color=INK, lw=2.6, ls=(0, (6, 3)), zorder=3,
                        label=f"R$_2$eD ({aucs['R2eD']:.3f})")
        handles['R2eD'] = h
    fpr, tpr, _ = roc_curve(y, p)
    (h,) = axA.plot(fpr, tpr, color=HOPKINS_RED, lw=3.2, zorder=4,
                    label=f'{model} ({aucs[model]:.3f})')
    handles[model] = h

    axA.plot([0, 1], [0, 1], ':', color='#C4C4C4', lw=1.2, zorder=0)
    axA.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axA.set_xlabel('1 \u2212 Specificity', fontsize=11)
    axA.set_ylabel('Sensitivity', fontsize=11)

    legend_order = sorted(handles, key=lambda m: aucs[m], reverse=True)
    axA.legend([handles[m] for m in legend_order],
               [handles[m].get_label() for m in legend_order],
               loc='lower right', fontsize=8.5, framealpha=0.92)
    axA.grid(True, alpha=0.2)
    axA.set_aspect('equal', adjustable='box')

    # B: SHAP 
    X, y_full = load_cohort()
    if X is not None and (len(y_full) != len(y) or y_full.sum() != y.sum()):
        raise SystemExit(
            f"Cohort mismatch: this script engineered {len(y_full)} patients "
            f"({int(y_full.sum())} ruptured), the results folder has {len(y)} "
            f"({int(y.sum())} ruptured).")
    if X is not None:
        import shap
        fold_df = pd.read_csv(folder / 'nested_cv_per_fold.csv')
        pl = fold_df[fold_df['Model'] == model]['Best_params'].tolist()
        params = json.loads(Counter(pl).most_common(1)[0][0]) if pl else {}
        final = build_model(model, params).fit(X, y_full)
        Xs = final.named_steps['scale'].transform(X)
        sv = shap.TreeExplainer(final.named_steps['clf']).shap_values(Xs)
        sv = np.asarray(sv, float)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        if sv.shape[1] == len(BASE_FEATURES) + 1:
            sv = sv[:, :-1]
        draw_beeswarm(axB, sv, X, BASE_FEATURES)
    else:
        axB.text(0.5, 0.5, 'Registry not readable\nset DATA_PATHS',
                 ha='center', va='center', transform=axB.transAxes,
                 fontsize=11, color=MUTED)
        axB.axis('off')

    # C: distribution 
    bins = np.linspace(0, 1, 41)
    for b in BANDS:
        m_ = band == b
        if m_.any():
            axC.hist(p[m_], bins=bins, color=BAND_COLORS[b], alpha=0.82,
                     edgecolor='white', linewidth=0.6,
                     label=f"{b}  (n={stats[b]['n']}, "
                           f"{100 * stats[b]['n'] / len(y):.0f}%)")
    for cut in (LOW_CUT, HIGH_CUT):
        axC.axvline(cut, color=INK, ls='--', lw=1.15, alpha=0.55)
    axC.set_xlabel('Predicted rupture probability (out-of-fold)', fontsize=11)
    axC.set_ylabel('Patients', fontsize=11)
    axC.legend(fontsize=9, frameon=False, loc='upper right')
    axC.grid(True, axis='y', alpha=0.16)
    axC.set_axisbelow(True)
    for sp in ('top', 'right'):
        axC.spines[sp].set_visible(False)

    # D: predicted vs observed 
    xp = np.arange(3)
    w = 0.36
    pred = [stats[b]['pred'] for b in BANDS]
    obs = [stats[b]['obs'] for b in BANDS]
    axD.bar(xp - w / 2, pred, w, color=[BAND_COLORS[b] for b in BANDS],
            alpha=0.45, edgecolor='white', label='Predicted')
    axD.bar(xp + w / 2, obs, w, color=[BAND_COLORS[b] for b in BANDS],
            alpha=0.95, edgecolor='white', label='Observed')
    for i, b in enumerate(BANDS):
        pr, ob = stats[b]['pred'], stats[b]['obs']
        if np.isnan(ob):
            continue
        axD.text(i - w / 2, pr + 1.6, f"{pr:.1f}%", ha='center', fontsize=9,
                 color=MUTED)
        axD.text(i + w / 2, ob + 1.6, f"{ob:.1f}%", ha='center', fontsize=9,
                 color=INK, fontweight='bold')
    axD.set_xticks(xp)
    axD.set_xticklabels([f"{b} risk\nn = {stats[b]['n']}" for b in BANDS],
                        fontsize=10)
    axD.set_ylim(0, 78)
    axD.set_ylabel('Rupture rate (%)', fontsize=11)
    axD.legend(fontsize=9, frameon=False, loc='upper left')
    axD.grid(True, axis='y', alpha=0.16)
    axD.set_axisbelow(True)
    for sp in ('top', 'right'):
        axD.spines[sp].set_visible(False)
    axD.tick_params(axis='x', length=0, pad=8)
    for ax, letter, title in [
            (axA, 'A', 'Discrimination'),
            (axB, 'B', f'Feature contributions - {model} (SHAP)'),
            (axC, 'C', 'Distribution of predicted risk'),
            (axD, 'D', 'Predicted vs observed rupture rate')]:
        ax.set_title(f'{letter}. {title}', fontsize=12.5, fontweight='bold',
                     loc='left', pad=10)

    out = folder / f'FIGURE_main_{model.lower().replace(" ", "_")}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    gap = np.nansum([abs(stats[b]['pred'] - stats[b]['obs']) * stats[b]['n']
                     for b in BANDS]) / len(y)
    print(f"{model}: AUC {aucs[model]:.4f}, Brier {brier_score_loss(y, p):.4f}, "
          f"mean band gap {gap:.2f} pts")
    print(f"Saved {out}")
    print(f"Saved {str(out).replace('.png', '.pdf')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
