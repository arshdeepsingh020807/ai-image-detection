"""Classification metrics, temperature scaling and the inconclusive band. Positive class = AI-generated."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over equal-width confidence bins."""
    confidence = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "n": len(y),
        "threshold": threshold,
        "accuracy": _safe_div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "roc_auc": float(roc_auc_score(y, prob)) if both else None,
        "pr_auc": float(average_precision_score(y, prob)) if both else None,
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
        "ece": expected_calibration_error(prob, y),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T > 0 minimising the negative log-likelihood of sigmoid(logit / T)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y).astype(int)

    def nll(log_t: float) -> float:
        p = np.clip(expit(logits / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def choose_band(prob: np.ndarray, y: np.ndarray, max_error: float = 0.02) -> tuple[float, float]:
    """Pick (low, high): p <= low -> real, p >= high -> AI, in between -> inconclusive.

    `high` is the smallest cut keeping the false-positive rate on real images <= max_error and `low` the
    largest cut keeping the false-negative rate on AI images <= max_error. If they cross, fall back to 0.5.
    """
    prob, y = np.asarray(prob, dtype=float), np.asarray(y).astype(int)
    real, ai = np.sort(prob[y == 0]), np.sort(prob[y == 1])
    high = float(np.quantile(real, 1 - max_error, method="higher")) if len(real) else 0.5
    low = float(np.quantile(ai, max_error, method="lower")) if len(ai) else 0.5
    low, high = min(low, 0.5), max(high, 0.5)
    return low, high


def band_metrics(prob: np.ndarray, y: np.ndarray, low: float, high: float) -> dict:
    prob, y = np.asarray(prob, dtype=float), np.asarray(y).astype(int)
    decided = (prob <= low) | (prob >= high)
    pred = (prob >= high).astype(int)
    errors = decided & (pred != y)
    real_fp = ((pred == 1) & decided & (y == 0)).sum()
    return {
        "coverage": float(decided.mean()) if len(y) else 0.0,
        "accuracy_on_decided": _safe_div(int((decided & (pred == y)).sum()), int(decided.sum())),
        "false_positive_rate_on_real": _safe_div(int(real_fp), int((y == 0).sum())),
        "error_rate": float(errors.mean()) if len(y) else 0.0,
        "inconclusive": int((~decided).sum()),
    }
