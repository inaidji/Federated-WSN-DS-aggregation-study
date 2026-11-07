"""Offline metric suite for binary attack detection (attack = class 1).

Everything is computed from labels y and attack probabilities p, pooled over the
clients' test sets. Includes the post-hoc calibration controls (temperature
scaling and Platt scaling) that are fitted on a held-out validation set only.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

EPS = 1e-7


def _clip(p):
    return np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.5), EPS, 1 - EPS)


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def log_loss_np(y, p):
    p = _clip(p)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece_binary(y, p, bins=15):
    """Expected calibration error of the attack probability (equal-width bins)."""
    y, p = np.asarray(y), _clip(p)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability_bins(y, p, bins=10):
    y, p = np.asarray(y), _clip(p)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append((float(p[m].mean()) if m.any() else np.nan,
                    float(y[m].mean()) if m.any() else np.nan, int(m.sum())))
    return out


def tpr_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


def all_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = _clip(p)
    pred = (p > thr).astype(int)
    pr, rc, f1a, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_acc=float(balanced_accuracy_score(y, pred)),
        atk_precision=float(pr[0]), atk_recall=float(rc[0]), atk_f1=float(f1a[0]),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
        log_loss=log_loss_np(y, p), brier=float(brier_score_loss(y, p)), ece=ece_binary(y, p),
        tpr_at_fpr1=tpr_at_fpr(y, p, 0.01), tpr_at_fpr5=tpr_at_fpr(y, p, 0.05),
    )


def fit_temperature(y, p):
    z, y = logit(p), np.asarray(y)

    def nll(logT):
        q = _clip(1 / (1 + np.exp(-z / np.exp(logT))))
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    T = float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))
    return (lambda p2: 1 / (1 + np.exp(-logit(p2) / T))), T


def fit_platt(y, p):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p).reshape(-1, 1), np.asarray(y))
    return (lambda p2: lr.predict_proba(logit(p2).reshape(-1, 1))[:, 1]), (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
