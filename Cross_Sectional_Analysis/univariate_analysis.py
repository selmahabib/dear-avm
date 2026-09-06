"""
Univariate Analysis
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm as _stdnorm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from stats_helpers import _benjamini_hochberg

def _is_binary(col):
    u = pd.Series(col).dropna().unique()
    try:
        return len(u) <= 2 and set(np.unique(u).astype(int)) <= {0, 1}
    except Exception:
        return False

def _univariate_logreg(x, y_):
    x = np.asarray(x, dtype=float).ravel()
    y_ = np.asarray(y_, dtype=int).ravel()
    X_d = np.column_stack([np.ones_like(x), x])
    m = LogisticRegression(C=1e12, solver='lbfgs', max_iter=5000, fit_intercept=False)
    m.fit(X_d, y_)
    coef = m.coef_.ravel()
    eta = X_d @ coef
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    XtWX = X_d.T @ (X_d * W[:, None])
    try:
        cov = np.linalg.inv(XtWX)
    except Exception:
        return (np.nan,) * 4
    se = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.nan
    beta = float(coef[1])
    if not np.isfinite(se) or se == 0:
        return np.exp(beta), np.nan, np.nan, np.nan
    z = beta / se
    return (float(np.exp(beta)), float(np.exp(beta - 1.96 * se)),
            float(np.exp(beta + 1.96 * se)), float(2 * (1 - _stdnorm.cdf(abs(z)))))

def run_univariate(clean_df, y, features, out_csv='univariate_analysis.csv'):
    """Univariate table for the candidate predictors.
    Returns the sorted DataFrame and writes it to out_csv.
    """
    univar_rows = []
    for feat in features:
        if feat not in clean_df.columns:
            continue
        col = clean_df[feat]
        binary = _is_binary(col)
        n_pos_u = int((y == 1).sum())
        n_neg_u = int((y == 0).sum())
        if binary:
            v = col.astype(int).values
            n11 = int(((v == 1) & (y == 1)).sum())
            n10 = int(((v == 0) & (y == 1)).sum())
            n01 = int(((v == 1) & (y == 0)).sum())
            n00 = int(((v == 0) & (y == 0)).sum())
            rupt_str = f"{n11}/{n_pos_u} ({100 * n11 / n_pos_u:.1f}%)"
            unr_str = f"{n01}/{n_neg_u} ({100 * n01 / n_neg_u:.1f}%)"
            try:
                _, p_test = fisher_exact(np.array([[n11, n10], [n01, n00]]))
                test_name = "Fisher"
            except Exception:
                p_test, test_name = np.nan, "-"
            OR, lo, hi, p_lr = _univariate_logreg(col.values, y)
            OR_label = 'OR'
        else:
            x_pos = col[y == 1].values
            x_neg = col[y == 0].values
            rupt_str = f"{x_pos.mean():.2f} +/- {x_pos.std(ddof=1):.2f}"
            unr_str = f"{x_neg.mean():.2f} +/- {x_neg.std(ddof=1):.2f}"
            try:
                _, p_test = mannwhitneyu(x_pos, x_neg, alternative='two-sided')
                test_name = "MWU"
            except Exception:
                p_test, test_name = np.nan, "-"
            OR, lo, hi, p_lr = _univariate_logreg(col.values, y)
            OR_label = 'OR (per unit)'
        try:
            a = roc_auc_score(y, col.values)
            auc_uni = a if a >= 0.5 else 1 - a
        except Exception:
            auc_uni = np.nan
        univar_rows.append({
            'Feature': feat, 'Type': 'binary' if binary else 'continuous',
            f'Ruptured (n={n_pos_u})': rupt_str, f'Unruptured (n={n_neg_u})': unr_str,
            'Test': test_name, 'p_test': p_test,
            OR_label: OR, 'OR_95CI_low': lo, 'OR_95CI_high': hi,
            'p_logreg_Wald': p_lr, 'AUC_univariate': auc_uni})
    univar_df = pd.DataFrame(univar_rows)
    or_col = next((c for c in univar_df.columns if c == 'OR' or c.startswith('OR (')), None)
    univar_df['q_logreg_BH'] = _benjamini_hochberg(univar_df['p_logreg_Wald'].values)
    univar_df = univar_df.sort_values('p_logreg_Wald').reset_index(drop=True)
    univar_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return univar_df
