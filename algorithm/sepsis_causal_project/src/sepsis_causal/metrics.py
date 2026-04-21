from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def safe_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    out: dict[str, float] = {}

    if np.unique(y_true).size < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(np.int64)
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def probability_error_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    mae = float(np.mean(np.abs(y_prob - y_true)))
    rmse = float(np.sqrt(np.mean((y_prob - y_true) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
    }


def treatment_effect_metrics(
    pred_all: np.ndarray,
    true_all: np.ndarray,
) -> dict[str, float]:
    # pred_all/true_all: [N, C], C=18
    if pred_all.ndim != 2 or true_all.ndim != 2:
        raise ValueError("pred_all and true_all must be 2D arrays")
    if pred_all.shape != true_all.shape:
        raise ValueError("pred_all and true_all shape mismatch")

    base = 0
    tau_pred = pred_all[:, 1:] - pred_all[:, [base]]
    tau_true = true_all[:, 1:] - true_all[:, [base]]

    pehe = math.sqrt(float(np.mean((tau_pred - tau_true) ** 2)))
    ate_pred = tau_pred.mean(axis=0)
    ate_true = tau_true.mean(axis=0)
    ate_error = float(np.mean(np.abs(ate_pred - ate_true)))

    rec_idx = np.argmin(pred_all, axis=1)
    rec_true = true_all[np.arange(true_all.shape[0]), rec_idx]
    best_true = true_all.min(axis=1)
    regret = float(np.mean(rec_true - best_true))

    return {
        "pehe": pehe,
        "ate_error": ate_error,
        "policy_regret": regret,
    }
