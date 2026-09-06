"""
AVM RUPTURE AT PRESENTATION - Cross-Sectional Analysis

OUTPUTS

           roc_all_models_nested.png
           shap_beeswarm_*.png
           patient_risk_probability_*.png
           fold_stability_boxplot.png
           sensitivity_specificity_by_model.png
           delong_heatmap*.png

  Tables   summary_table.csv               
           clinical_metrics_*.csv
           bootstrap_ci_metrics_*.csv
           calibration_summary.csv
           risk_bands_*.csv
           univariate_analysis.csv
           nested_cv_per_fold.csv
           nested_oof_predictions.csv
           delong_pvalues.csv, delong_qvalues_BH.csv
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import os
import glob
import json
from collections import Counter
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import shap
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from stats_helpers import (                    
    _benjamini_hochberg, delong_roc_test,
    bootstrap_metric, auc_metric, avgprec_metric,
    brier_metric, lock_youden, lock_sens_target, counts_at,
    metrics_from_counts, hosmer_lemeshow, calib_slope_intercept,
    spiegelhalter_z,
)
from univariate_analysis import run_univariate  
from generate_csv_and_plots import (          
    make_roc_figure, make_shap_beeswarm, make_risk_band_figure,
    make_sens_spec_figure, make_fold_stability_figure, make_delong_heatmap,
)

np.random.seed(42)
RANDOM_STATE = 42
# Settings
N_OUTER = 10
N_INNER = 5
N_BOOT = 2000
N_JOBS = -1
SHAP_MODELS = ['CatBoost']   # which models get SHAP figures
CLASS_WEIGHTING = False
CALIBRATION = None
CALIBRATION_CV = 5
RISK_BANDS = (0.20, 0.50)  # low / medium / high cutoffs for the risk figure

# Load Data
data_paths = [
    r'path/to/your/data.xlsx'
]
df_raw = None
for path in data_paths:
    if os.path.exists(path):
        df_raw = pd.read_excel(path)
        print(f"Loaded {len(df_raw)} patients from {path}")
        _src = os.path.dirname(os.path.abspath(path))
        break
if df_raw is None:
    raise FileNotFoundError(
        "Registry not found - check data_paths at the top of this file.")

RESULTS_DIR = os.path.join(
    _src, f"bavm_cross_sectional_analysis_{datetime.now():%Y%m%d_%H%M}")
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '.last_run'), 'w') as fh:
    fh.write(RESULTS_DIR)
os.chdir(RESULTS_DIR)

# Feature Engineering
print("FEATURE ENGINEERING:")
df = df_raw.copy()
df['Race_nonwhite'] = (df['Race'] != '1_White').astype(float)
df.loc[df['Race'].isna(), 'Race_nonwhite'] = np.nan
df['Age'] = pd.to_numeric(df['Age_dx'], errors='coerce')
deep_cols = ['Location_brainstem', 'Location_basal_ganglia', 'Location_thalamus',
             'Location_Cerebellar_hemisphere', 'Location_Cerebellar_vermis']
existing_deep = [c for c in deep_cols if c in df.columns]
if existing_deep:
    _deep = df[existing_deep]
    df['Deep_Location'] = _deep.any(axis=1).astype(float)
    df.loc[_deep.isna().all(axis=1), 'Deep_Location'] = np.nan

df['Size'] = pd.to_numeric(df['max_Size'], errors='coerce')
df['Size_less_3cm'] = (df['Size'] < 3).astype(float)
df.loc[df['Size'].isna(), 'Size_less_3cm'] = np.nan
if {'Drainage_vein_deep', 'Drainage_vein_superficial'} <= set(df.columns):
    _dd = df[['Drainage_vein_deep', 'Drainage_vein_superficial']]
    df['Exclusive_Deep_Drainage'] = (
        (df['Drainage_vein_deep'] == 1) & (df['Drainage_vein_superficial'] == 0)
    ).astype(float)
    df.loc[_dd.isna().any(axis=1), 'Exclusive_Deep_Drainage'] = np.nan
# Feeder count and we consider Undocumented as a separate state
for col_name in ['Feeder_Territory_Number', 'Feeder_Teriitory_Number']:
    if col_name in df.columns:
        _ft = pd.to_numeric(df[col_name], errors='coerce')
        df['Single_Feeding_Artery'] = (_ft == 1).astype(float)
        df['Feeder_Count_Unknown'] = _ft.isna().astype(float)
        df.loc[_ft.isna(), 'Single_Feeding_Artery'] = 0.0
        _n_unk = int(df['Feeder_Count_Unknown'].sum())
        print(f"  Feeder_Count_Unknown: {_n_unk} patients "
              f"({100 * _n_unk / len(df):.1f}%) kept as an explicit level")
        break
if 'Aneurysm_intranidal' in df.columns:
    df['Intranidal_Aneurysm'] = pd.to_numeric(df['Aneurysm_intranidal'], errors='coerce')
if 'Hypertension' in df.columns:
    df['HTN'] = pd.to_numeric(df['Hypertension'], errors='coerce')
df['Event'] = pd.to_numeric(df['Ever_Ruptured'], errors='coerce')
BASE_FEATURES = [f for f in [
    'Race_nonwhite', 'Deep_Location', 'Size', 'Exclusive_Deep_Drainage',
    'Single_Feeding_Artery', 'Feeder_Count_Unknown', 'Age',
    'Intranidal_Aneurysm', 'HTN'] if f in df.columns]
clean_df = df[BASE_FEATURES + ['Size_less_3cm', 'Event']].dropna().copy()
clean_df['Event'] = clean_df['Event'].astype(int)

R2ED_FEATURES = [f for f in ['Race_nonwhite', 'Deep_Location',
                             'Exclusive_Deep_Drainage', 'Single_Feeding_Artery']
                 if f in clean_df.columns] + ['Size_less_3cm']
X = clean_df[BASE_FEATURES].values.astype(float)
X_r2ed = clean_df[R2ED_FEATURES].values.astype(float)
y = clean_df['Event'].values.astype(int)

print(f"Model features ({len(BASE_FEATURES)}): {BASE_FEATURES}")
print(f"  R2eD features  ({len(R2ED_FEATURES)}): {R2ED_FEATURES}")
print(f"  Cohort {len(clean_df)} | ruptured {y.sum()} ({100 * y.mean():.1f}%) "
      f"| dropped {len(df_raw) - len(clean_df)}")

# Helpers
# Operating points
#   Which cutoffs to lock. Youden plus the sensitivity targets that matter
#   clinically.
OPERATING_POINTS = {
    'Youden':      lambda y_, p_: lock_youden(y_, p_),
    'Sens >= 65%': lambda y_, p_: lock_sens_target(y_, p_, 0.65),
    'Sens >= 70%': lambda y_, p_: lock_sens_target(y_, p_, 0.70),
}
OP_TAGS = {'Youden': 'youden', 'Sens >= 65%': 'sens65', 'Sens >= 70%': 'sens70'}

# Univariate Analysis
print("UNIVARIATE ANALYSIS:")
univar_df = run_univariate(clean_df, y, BASE_FEATURES + ['Size_less_3cm'])

# Models
def build_estimators(y_fit):
    n_pos = max(int((y_fit == 1).sum()), 1)
    spw = (int((y_fit == 0).sum()) / n_pos) if CLASS_WEIGHTING else 1.0
    cw = 'balanced' if CLASS_WEIGHTING else None
    cb_cw = 'Balanced' if CLASS_WEIGHTING else None

    defs = {
        'Logistic Regression': (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                               class_weight=cw),
            {'clf__C': [0.01, 0.1, 1.0, 10.0]}),
        'SVM': (
            SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE,
                class_weight=cw),
            {'clf__C': [0.1, 1.0, 10.0], 'clf__gamma': ['scale', 0.05]}),
        'Random Forest': (
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                   class_weight=cw, n_jobs=1),
            {'clf__max_depth': [4, 8, None], 'clf__min_samples_leaf': [1, 5]}),
        'AdaBoost': (
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {'clf__n_estimators': [100, 300], 'clf__learning_rate': [0.5, 1.0]}),
        'CatBoost': (
            cb.CatBoostClassifier(iterations=300, random_state=RANDOM_STATE,
                                  verbose=False, auto_class_weights=cb_cw,
                                  allow_writing_files=False),
            {'clf__depth': [3, 5], 'clf__learning_rate': [0.03, 0.1]}),
        'LightGBM': (
            lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE,
                               verbose=-1, class_weight=cw),
            {'clf__num_leaves': [7, 15, 31], 'clf__learning_rate': [0.03, 0.1]}),
        'XGBoost': (
            xgb.XGBClassifier(n_estimators=300, random_state=RANDOM_STATE,
                              eval_metric='logloss', scale_pos_weight=spw),
            {'clf__max_depth': [3, 5], 'clf__learning_rate': [0.03, 0.1]}),
    }
    return {name: (Pipeline([('scale', StandardScaler()), ('clf', est)]), grid)
            for name, (est, grid) in defs.items()}


def wrap_calibration(estimator):
    if not CALIBRATION:
        return estimator
    return CalibratedClassifierCV(estimator=clone(estimator),
                                  method=CALIBRATION, cv=CALIBRATION_CV)


def r2ed_pipeline():
    return Pipeline([('scale', StandardScaler()),
                     ('clf', LogisticRegression(
                         max_iter=1000, random_state=RANDOM_STATE,
                         class_weight='balanced' if CLASS_WEIGHTING else None))])



# Nested Cross-Validation
print(f"NESTED CROSS-VALIDATION ({N_OUTER} outer x {N_INNER} inner):")
outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

MODEL_NAMES = list(build_estimators(y).keys())
ALL_NAMES = MODEL_NAMES + ['R2eD']

oof = {n: np.full(len(y), np.nan) for n in ALL_NAMES}
fold_auc = {n: [] for n in ALL_NAMES}
pooled = {n: {op: dict(TP=0, FP=0, FN=0, TN=0) for op in OPERATING_POINTS}
          for n in ALL_NAMES}
thr_per_fold = {n: {op: [] for op in OPERATING_POINTS} for n in ALL_NAMES}
params_per_fold = {n: [] for n in MODEL_NAMES}
fold_rows = []

for k, (tr, te) in enumerate(outer.split(X, y), start=1):
    print(f"\n  fold {k}/{N_OUTER}  (train {len(tr)}, test {len(te)}, "
          f"{int(y[te].sum())} ruptured)")

    jobs = list(build_estimators(y[tr]).items()) + [('R2eD', (r2ed_pipeline(), {}))]

    for name, (pipe, grid) in jobs:
        Xtr, Xte = (X_r2ed[tr], X_r2ed[te]) if name == 'R2eD' else (X[tr], X[te])

        if grid:
            gs = GridSearchCV(pipe, grid, scoring='roc_auc', cv=inner,
                              n_jobs=N_JOBS, refit=True)
            gs.fit(Xtr, y[tr])
            chosen, params = gs.best_estimator_, gs.best_params_
            params_per_fold[name].append(json.dumps(params, sort_keys=True))
        else:
            chosen, params = clone(pipe), {}
        best = wrap_calibration(chosen)
        best.fit(Xtr, y[tr])
        inner_oof = cross_val_predict(clone(best), Xtr, y[tr], cv=inner,
                                      method='predict_proba', n_jobs=N_JOBS)[:, 1]
        thr = {op: fn(y[tr], inner_oof) for op, fn in OPERATING_POINTS.items()}

        p_te = best.predict_proba(Xte)[:, 1]
        oof[name][te] = p_te
        a = roc_auc_score(y[te], p_te)
        fold_auc[name].append(a)

        for op in OPERATING_POINTS:
            c = counts_at(y[te], p_te, thr[op])
            for kk in pooled[name][op]:
                pooled[name][op][kk] += c[kk]
            thr_per_fold[name][op].append(thr[op])

        fold_rows.append({'Fold': k, 'Model': name, 'Outer_test_AUC': a,
                          **{f'Thr_{OP_TAGS[op]}': thr[op] for op in OPERATING_POINTS},
                          'Best_params': json.dumps(params, sort_keys=True)})
        print(f"     {name:<20} AUC={a:.3f}  {params if params else ''}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv('nested_cv_per_fold.csv', index=False)

oof_out = pd.DataFrame({n: oof[n] for n in ALL_NAMES})
oof_out['y_true'] = y
oof_out.to_csv('nested_oof_predictions.csv', index=False)
print("\nSaved nested_cv_per_fold.csv, nested_oof_predictions.csv")

nested_auc = {n: roc_auc_score(y, oof[n]) for n in ALL_NAMES}
RANKED = sorted(MODEL_NAMES, key=lambda n: nested_auc[n], reverse=True)
BEST_MODEL = RANKED[0]
print(f"\nBest model under nested CV: {BEST_MODEL} (AUC {nested_auc[BEST_MODEL]:.4f})")
print(f"R2eD: {nested_auc['R2eD']:.4f}")



# Clinical Metrics
print("CLINICAL METRICS:")
all_rows = []
for op in OPERATING_POINTS:
    for name in RANKED + ['R2eD']:
        thrs = thr_per_fold[name][op]
        note = f"{np.mean(thrs):.3f} (fold range {min(thrs):.3f}-{max(thrs):.3f})"
        all_rows.append(metrics_from_counts(name, op, pooled[name][op], y, oof[name], note))

clinical_full_df = pd.DataFrame(all_rows)
clinical_full_df.to_csv('clinical_metrics_full_table.csv', index=False)
for op, tag in OP_TAGS.items():
    sub = clinical_full_df[clinical_full_df['Operating_Point'] == op]
    sub = sub.sort_values('Sensitivity', ascending=False)
    print(f"\n--- {op} ---")
    print(sub.drop(columns=['Operating_Point']).to_string(index=False))
    sub.to_csv(f'clinical_metrics_{tag}.csv', index=False)
print("\nSaved clinical_metrics_*.csv")

# Bootstrap CIs
print(f"BOOTSTRAP 95% CIs (n={N_BOOT}, stratified):")
ci_tables = {}
for op, tag in OP_TAGS.items():
    rows = []
    for name in RANKED + ['R2eD']:
        p = oof[name]
        auc_med, (a_lo, a_hi) = bootstrap_metric(y, p, auc_metric)
        ap_med, (ap_lo, ap_hi) = bootstrap_metric(y, p, avgprec_metric)
        br_med, (b_lo, b_hi) = bootstrap_metric(y, p, brier_metric)
        c = pooled[name][op]
        rates = metrics_from_counts(name, op, c, y, p, '')
        rows.append({
            'Model': name, 'Operating_Point': op,
            'AUC':          f"{auc_med:.3f} [{a_lo:.3f}, {a_hi:.3f}]",
            'AvgPrecision': f"{ap_med:.3f} [{ap_lo:.3f}, {ap_hi:.3f}]",
            'Brier':        f"{br_med:.3f} [{b_lo:.3f}, {b_hi:.3f}]",
            'Sensitivity':  f"{rates['Sensitivity']:.1f}",
            'Specificity':  f"{rates['Specificity']:.1f}",
            'PPV':          f"{rates['PPV']:.1f}",
            'NPV':          f"{rates['NPV']:.1f}",
        })
    t = pd.DataFrame(rows)
    t.to_csv(f'bootstrap_ci_metrics_{tag}.csv', index=False)
    ci_tables[op] = t
    print(f"\n--- {op} ---")
    print(t.to_string(index=False))
print("Saved bootstrap_ci_metrics_*.csv")

# Calibration Metrics
print("CALIBRATION:")
cal_rows = []
for name in RANKED + ['R2eD']:
    p = oof[name]
    bri = brier_score_loss(y, p)
    pi = y.mean()
    chi2, dfc, p_hl = hosmer_lemeshow(y, p)
    inter, slope = calib_slope_intercept(y, p)
    z_sh, p_sh = spiegelhalter_z(y, p)
    cal_rows.append({'Model': name, 'Brier': bri,
                     'BrierSkillScore': 1 - bri / (pi * (1 - pi)),
                     'HL_chi2': chi2, 'HL_df': dfc, 'HL_p': p_hl,
                     'Calib_Intercept': inter, 'Calib_Slope': slope,
                     'Spiegelhalter_Z': z_sh, 'Spiegelhalter_p': p_sh})
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv('calibration_summary.csv', index=False)
print(cal_df.to_string(index=False, float_format='%.4f'))
print("Saved calibration_summary.csv")

# ROC FIGURE
make_roc_figure(y, oof, nested_auc, RANKED)
print("Saved roc_all_models_nested.png")

# SHAP
print("SHAP:")
if CALIBRATION:
    print(f"  Calibration is on ({CALIBRATION}); SHAP is computed on the "
          f"uncalibrated base model.\n  Calibration is monotone, so feature "
          f"ordering is unaffected.")

SHAP_TARGETS = list(dict.fromkeys([BEST_MODEL] + SHAP_MODELS))
if BEST_MODEL not in SHAP_MODELS:
    print(f"  note: {BEST_MODEL} won under nested CV, adding it to the SHAP list")

for name in SHAP_TARGETS:
    if name not in MODEL_NAMES:
        continue
    modal = Counter(params_per_fold[name]).most_common(1)[0]
    params = json.loads(modal[0])
    print(f"{name}: {params}  (selected in {modal[1]}/{N_OUTER} folds)")
    pipe, _ = build_estimators(y)[name]
    final = clone(pipe).set_params(**params).fit(X, y)
    Xs = final.named_steps['scale'].transform(X)
    est = final.named_steps['clf']

    try:
        sv = shap.TreeExplainer(est).shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        if sv.shape[1] == len(BASE_FEATURES) + 1:
            sv = sv[:, :-1]
        sv = np.asarray(sv, dtype=float)
    except Exception as e:
        print(f"  SHAP failed for {name}: {e}")
        continue

    tag = name.lower().replace(' ', '_')
    make_shap_beeswarm(sv, X, BASE_FEATURES, name, f'shap_beeswarm_{tag}.png')

    imp = np.abs(sv).mean(axis=0)
    print("  Top features by mean |SHAP|:")
    for i, (f_, v) in enumerate(sorted(zip(BASE_FEATURES, imp),
                                       key=lambda t: t[1], reverse=True)[:10], 1):
        print(f"    {i:>2}. {f_:<26} {v:.4f}")
    pd.DataFrame({'Feature': BASE_FEATURES, 'Mean_abs_SHAP': imp}) \
        .sort_values('Mean_abs_SHAP', ascending=False) \
        .to_csv(f'shap_importance_{tag}.csv', index=False)
    print(f"  Saved shap_beeswarm_{tag}.png, shap_importance_{tag}.csv")



# RISK STRATIFICATION 
for name in SHAP_TARGETS:
    if name not in MODEL_NAMES:
        continue
    tag = name.lower().replace(' ', '_')
    p = oof[name]
    lo_cut, hi_cut = RISK_BANDS

    bands = np.where(p < lo_cut, 'Low Risk',
                     np.where(p < hi_cut, 'Medium Risk', 'High Risk'))
    print(f"\nRisk bands - {name} (out-of-fold):")
    band_rows = []
    for cat in ['Low Risk', 'Medium Risk', 'High Risk']:
        m = bands == cat
        n = int(m.sum())
        if n == 0:
            continue
        print(f"  {cat:>12}: {n:>4} | predicted {100 * p[m].mean():>5.1f}% "
              f"| observed {100 * y[m].mean():>5.1f}%")
        band_rows.append({'Model': name, 'Band': cat, 'N': n,
                          'Mean_predicted_%': round(100 * p[m].mean(), 1),
                          'Observed_rupture_%': round(100 * y[m].mean(), 1)})
    pd.DataFrame(band_rows).to_csv(f'risk_bands_{tag}.csv', index=False)

    make_risk_band_figure(y, p, bands, ['Low Risk', 'Medium Risk', 'High Risk'],
                          lo_cut, hi_cut, name,
                          f'patient_risk_probability_{tag}.png')

    print(f"Saved patient_risk_probability_{tag}.png, risk_bands_{tag}.csv")

# Sensitivity and Specificity by model
make_sens_spec_figure(clinical_full_df, ['Sens >= 65%', 'Sens >= 70%'])
print("Saved sensitivity_specificity_by_model.png")

# Fold Stability
make_fold_stability_figure(fold_auc, ALL_NAMES, N_OUTER)
print("Saved fold_stability_boxplot.png")

# PAIRWISE DeLONG
print("Pairwise DeLong and BH-FDR:")
names_all = RANKED + ['R2eD']
N = len(names_all)
M = pd.DataFrame(np.ones((N, N)), index=names_all, columns=names_all)
pv, pairs = [], []
for a, b in combinations(names_all, 2):
    _, _, _, p = delong_roc_test(y, oof[a], oof[b])
    M.loc[a, b] = M.loc[b, a] = p
    pv.append(p)
    pairs.append((a, b))
M.to_csv('delong_pvalues.csv')
Q = pd.DataFrame(np.ones((N, N)), index=names_all, columns=names_all)
for (a, b), q in zip(pairs, _benjamini_hochberg(np.array(pv))):
    Q.loc[a, b] = Q.loc[b, a] = q
Q.to_csv('delong_qvalues_BH.csv')

for label, mat, fn in [('p-values', M, 'delong_heatmap.png'),
                       ('BH-FDR q-values', Q, 'delong_heatmap_BHFDR.png')]:
    make_delong_heatmap(mat, label, fn)
print("Saved delong_pvalues.csv, delong_qvalues_BH.csv, heatmaps")


# Summary Table
print("SUMMARY TABLE:")
# Baseline predictions which we used for every DeLong comparison below
p_ref = oof['R2eD']

ci_y = ci_tables['Youden'].set_index('Model').to_dict('index')
ci_65 = ci_tables['Sens >= 65%'].set_index('Model').to_dict('index')
ci_70 = ci_tables['Sens >= 70%'].set_index('Model').to_dict('index')
cal_lookup = cal_df.set_index('Model').to_dict('index')

master_rows = []
for name in RANKED + ['R2eD']:
    fa = np.array(fold_auc[name])
    cal_ = cal_lookup.get(name, {})
    if name != 'R2eD':
        _, _, _, p_d = delong_roc_test(y, oof[name], p_ref)
        delta = nested_auc[name] - nested_auc['R2eD']
    else:
        p_d, delta = np.nan, np.nan
    master_rows.append({
        'Model': name,
        'Nested_AUC': ci_y.get(name, {}).get('AUC', '-'),
        'Fold_AUC_mean_SD': f"{fa.mean():.3f} ({fa.std(ddof=1):.3f})",
        'Fold_AUC_range': f"{fa.min():.3f}-{fa.max():.3f}",
        'AUPRC': ci_y.get(name, {}).get('AvgPrecision', '-'),
        'Brier': f"{cal_.get('Brier', np.nan):.3f}",
        'HL_p': f"{cal_.get('HL_p', np.nan):.3f}",
        'Calib_slope': f"{cal_.get('Calib_Slope', np.nan):.2f}",
        'Calib_intercept': f"{cal_.get('Calib_Intercept', np.nan):.2f}",
        'Delta_AUC_vs_R2eD': f"{delta:+.4f}" if not np.isnan(delta) else '-',
        'DeLong_p_vs_R2eD': f"{p_d:.4g}" if not np.isnan(p_d) else '-',
        'Sens_Youden': ci_y.get(name, {}).get('Sensitivity', '-'),
        'Spec_Youden': ci_y.get(name, {}).get('Specificity', '-'),
        'Sens_S65': ci_65.get(name, {}).get('Sensitivity', '-'),
        'Spec_S65': ci_65.get(name, {}).get('Specificity', '-'),
        'Sens_S70': ci_70.get(name, {}).get('Sensitivity', '-'),
        'Spec_S70': ci_70.get(name, {}).get('Specificity', '-'),
    })
master_df = pd.DataFrame(master_rows)
master_df.to_csv('summary_table.csv', index=False)
print("Saved summary_table.csv")


best_fa = np.array(fold_auc[BEST_MODEL])
_, _, _, p_best_delong = delong_roc_test(y, oof[BEST_MODEL], p_ref)

print(f"\n{BEST_MODEL}: AUC {nested_auc[BEST_MODEL]:.4f} "
      f"{ci_y[BEST_MODEL]['AUC']}, per-fold SD {best_fa.std(ddof=1):.3f}, "
      f"vs R2eD {nested_auc[BEST_MODEL] - nested_auc['R2eD']:+.4f} "
      f"(DeLong p={p_best_delong:.4g})")
print(f"{len(glob.glob('*.png')) + len(glob.glob('*.csv'))} files in {RESULTS_DIR}")