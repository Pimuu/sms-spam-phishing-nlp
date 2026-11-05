"""Evaluation metrics, including bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def point_metrics(y_true, y_pred, y_score) -> dict:
    """Threshold metrics (spam = positive class) plus ranking metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        # False-positive rate = share of legitimate messages flagged as spam;
        # the operationally most expensive error for a message filter.
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_pred, y_score, n_boot: int = config.N_BOOTSTRAP,
                 seed: int = config.SEED, alpha: float = 0.05) -> dict:
    """Percentile bootstrap 95% CIs for F1 and ROC-AUC on the test set.

    Test messages are resampled with replacement (stratified by class so that
    every resample contains both classes). The same ``seed`` is used for every
    model, so all models are evaluated on identical resamples, which makes the
    intervals comparable across rows.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    f1s, aucs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        f1s[b] = f1_score(y_true[idx], y_pred[idx], zero_division=0)
        aucs[b] = roc_auc_score(y_true[idx], y_score[idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "f1_ci_low": float(np.percentile(f1s, lo)), "f1_ci_high": float(np.percentile(f1s, hi)),
        "roc_auc_ci_low": float(np.percentile(aucs, lo)), "roc_auc_ci_high": float(np.percentile(aucs, hi)),
    }


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test for two classifiers on the same test set.

    ``b`` = messages only A gets right, ``c`` = messages only B gets right.
    Under H0 (equal error rates) b ~ Binomial(b + c, 0.5). Returns the
    two-sided p-value. Uses only the standard library (math.comb).
    """
    from math import comb

    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_selected_correct": b, "only_other_correct": c, "p_value": p}
