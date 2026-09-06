"""
Statistical helpers for the cross-sectional analysis
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    matthews_corrcoef, roc_auc_score, roc_curve,
)
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
N_BOOT = 2000

def _benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = len(pm)
    if n == 0:
        return p.copy()
    order = np.argsort(pm)
    ranked = pm[order]
    q = np.empty_like(pm)
    cmin = 1.0
    for i in range(n - 1, -1, -1):
        cmin = min(cmin, ranked[i] * n / (i + 1))
        q[order[i]] = cmin
    out = p.copy()
    out[mask] = np.clip(q, 0, 1)
    return out

def delong_roc_test(y_true, pred1, pred2):
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.0, np.nan, np.nan, np.nan
    pos1, neg1 = pred1[y_true == 1], pred1[y_true == 0]
    pos2, neg2 = pred2[y_true == 1], pred2[y_true == 0]
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    V10_1 = np.array([np.mean(p > neg1) + 0.5 * np.mean(p == neg1) for p in pos1])
    V01_1 = np.array([np.mean(pos1 > n) + 0.5 * np.mean(pos1 == n) for n in neg1])
    V10_2 = np.array([np.mean(p > neg2) + 0.5 * np.mean(p == neg2) for p in pos2])
    V01_2 = np.array([np.mean(pos2 > n) + 0.5 * np.mean(pos2 == n) for n in neg2])
    S = (np.cov(np.column_stack([V10_1, V10_2]), rowvar=False) / n1
         + np.cov(np.column_stack([V01_1, V01_2]), rowvar=False) / n0)
    diff = auc1 - auc2
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return diff, auc1, auc2, np.nan
    z = diff / np.sqrt(var)
    return z, auc1, auc2, float(2 * sp_stats.norm.sf(abs(z)))

def stratified_boot_idx(y_arr, rng=RNG):
    pos = np.where(y_arr == 1)[0]
    neg = np.where(y_arr == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])

def bootstrap_metric(y_true, y_score, metric_fn, n_boot=N_BOOT, rng=RNG):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = []
    for _ in range(n_boot):
        idx = stratified_boot_idx(y_true, rng)
        try:
            out.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            out.append(np.nan)
    out = np.array(out)
    out = out[~np.isnan(out)]
    if len(out) == 0:
        return np.nan, (np.nan, np.nan)
    return float(np.median(out)), (float(np.percentile(out, 2.5)),
                                   float(np.percentile(out, 97.5)))

def auc_metric(y_, p_):
    return roc_auc_score(y_, p_)

def avgprec_metric(y_, p_):
    return average_precision_score(y_, p_)

def brier_metric(y_, p_):
    return brier_score_loss(y_, p_)

def lock_youden(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[np.argmax(tpr - fpr)])

def lock_sens_target(y_true, y_score, target):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = tpr >= target
    if not ok.any():
        return float(thr[np.argmax(tpr)])
    idx = np.where(ok)[0]
    return float(thr[idx[np.argmin(fpr[idx])]])

def counts_at(y_true, y_score, thr):
    yp = (np.asarray(y_score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), yp, labels=[0, 1]).ravel()
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

def metrics_from_counts(name, op_name, c, y_true, y_prob, thr_note):
    tp, fp, fn, tn = c['TP'], c['FP'], c['FN'], c['TN']
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    lr_pos = sens / (1 - spec) if (not np.isnan(spec) and spec < 1) else np.inf
    lr_neg = (1 - sens) / spec if (not np.isnan(spec) and spec > 0) else np.inf
    dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf
    y_pred = np.concatenate([np.ones(tp + fp), np.zeros(tn + fn)])
    y_obs = np.concatenate([np.ones(tp), np.zeros(fp), np.zeros(tn), np.ones(fn)])
    try:
        mcc = matthews_corrcoef(y_obs, y_pred)
    except Exception:
        mcc = np.nan
    prec = ppv
    rec = sens
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else np.nan
    return {
        'Model': name, 'Operating_Point': op_name,
        'N_evaluated': len(y_true), 'N_ruptured': int(np.sum(y_true)),
        'Threshold': thr_note,
        'AUC':         round(roc_auc_score(y_true, y_prob), 3),
        'AUPRC':       round(average_precision_score(y_true, y_prob), 3),
        'Brier':       round(brier_score_loss(y_true, y_prob), 3),
        'Sensitivity': round(sens * 100, 1) if not np.isnan(sens) else np.nan,
        'Specificity': round(spec * 100, 1) if not np.isnan(spec) else np.nan,
        'PPV':         round(ppv * 100, 1) if not np.isnan(ppv) else np.nan,
        'NPV':         round(npv * 100, 1) if not np.isnan(npv) else np.nan,
        'LR+':         round(lr_pos, 2) if lr_pos != np.inf else 'inf',
        'LR-':         round(lr_neg, 3) if lr_neg != np.inf else 'inf',
        'DOR':         round(dor, 1) if dor != np.inf else 'inf',
        'F1':          round(f1, 3) if not np.isnan(f1) else np.nan,
        'F2':          round(f2, 3) if not np.isnan(f2) else np.nan,
        'Accuracy':    round(100 * (tp + tn) / (tp + tn + fp + fn), 1),
        'MCC':         round(mcc, 3) if not np.isnan(mcc) else np.nan,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }


# Calibration helpers
def hosmer_lemeshow(y_true, y_prob, groups=10):
    d = pd.DataFrame({'y': y_true, 'p': y_prob})
    try:
        d['g'] = pd.qcut(d['p'], groups, labels=False, duplicates='drop')
    except Exception:
        d['g'] = pd.cut(d['p'], groups, labels=False, include_lowest=True)
    g = d.groupby('g').agg(o=('y', 'sum'), n=('y', 'count'), e=('p', 'sum'))
    g['e'] = g['e'].clip(lower=1e-9, upper=g['n'] - 1e-9)
    chi2 = ((g['o'] - g['e']) ** 2 / (g['e'] * (1 - g['e'] / g['n']))).sum()
    dfc = max(g.shape[0] - 2, 1)
    return float(chi2), dfc, float(1 - sp_stats.chi2.cdf(chi2, dfc))

def calib_slope_intercept(y_true, y_prob):
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    Xd = np.column_stack([np.ones_like(logit), logit])
    m = LogisticRegression(C=1e12, solver='lbfgs', max_iter=5000, fit_intercept=False)
    m.fit(Xd, y_true)
    c = m.coef_.ravel()
    return float(c[0]), float(c[1])

def spiegelhalter_z(y_true, y_prob):
    num = (y_true - y_prob).sum()
    den = np.sqrt(((1 - 2 * y_prob) ** 2 * y_prob * (1 - y_prob)).sum())
    if den == 0:
        return np.nan, np.nan
    z = num / den
    return float(z), float(2 * sp_stats.norm.sf(abs(z)))

