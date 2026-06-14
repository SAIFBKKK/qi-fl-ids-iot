"""Metrics helpers for P10 robustness reports."""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    counts = binary_confusion(y_true, y_pred)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision_attack = tp / max(tp + fp, 1)
    recall_attack = tp / max(tp + fn, 1)
    f1_attack = 2 * precision_attack * recall_attack / max(precision_attack + recall_attack, 1e-12)
    precision_normal = tn / max(tn + fn, 1)
    recall_normal = tn / max(tn + fp, 1)
    f1_normal = 2 * precision_normal * recall_normal / max(precision_normal + recall_normal, 1e-12)
    macro_f1 = (f1_normal + f1_attack) / 2.0
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    result: dict[str, Any] = {
        **counts,
        "accuracy": float(accuracy),
        "precision_attack": float(precision_attack),
        "attack_recall": float(recall_attack),
        "recall_attack": float(recall_attack),
        "f1_attack": float(f1_attack),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(((tn + fp) * f1_normal + (tp + fn) * f1_attack) / max(tp + tn + fp + fn, 1)),
        "FPR": float(fpr),
        "FNR": float(fnr),
    }
    return result


def robustness_score(metrics: dict[str, Any]) -> float:
    return float(0.5 * metrics.get("macro_f1", 0.0) + 0.3 * metrics.get("attack_recall", 0.0) - 0.2 * metrics.get("FPR", 0.0))
