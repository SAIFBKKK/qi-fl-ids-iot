"""Metric helpers for P8 QGA and reduced-feature adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from models.metrics import binary_metrics, classification_report_dict, predictions_from_threshold
except Exception:  # pragma: no cover - fallback for isolated import tests
    def predictions_from_threshold(prob_attack: np.ndarray, threshold: float) -> np.ndarray:
        return (np.asarray(prob_attack) >= threshold).astype(np.int64)

    def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        div = lambda a, b: float(a / b) if b else 0.0
        precision_attack = div(tp, tp + fp)
        recall_attack = div(tp, tp + fn)
        f1_attack = div(2 * precision_attack * recall_attack, precision_attack + recall_attack)
        precision_normal = div(tn, tn + fn)
        recall_normal = div(tn, tn + fp)
        f1_normal = div(2 * precision_normal * recall_normal, precision_normal + recall_normal)
        return {
            "accuracy": div(tp + tn, tp + tn + fp + fn),
            "macro_f1": (f1_normal + f1_attack) / 2,
            "weighted_f1": (f1_normal * (tn + fp) + f1_attack * (tp + fn)) / max(tp + tn + fp + fn, 1),
            "precision_attack": precision_attack,
            "recall_attack": recall_attack,
            "f1_attack": f1_attack,
            "FPR": div(fp, fp + tn),
            "FNR": div(fn, fn + tp),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }

    def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        return {"binary": binary_metrics(y_true, y_pred)}


def metrics_from_probabilities(y_true: np.ndarray, prob_attack: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = predictions_from_threshold(prob_attack, threshold)
    return binary_metrics(y_true, y_pred)


def confusion_matrix_rows(metrics: dict[str, Any]) -> list[dict[str, int]]:
    return [
        {"actual": 0, "predicted": 0, "count": int(metrics.get("TN", 0))},
        {"actual": 0, "predicted": 1, "count": int(metrics.get("FP", 0))},
        {"actual": 1, "predicted": 0, "count": int(metrics.get("FN", 0))},
        {"actual": 1, "predicted": 1, "count": int(metrics.get("TP", 0))},
    ]
