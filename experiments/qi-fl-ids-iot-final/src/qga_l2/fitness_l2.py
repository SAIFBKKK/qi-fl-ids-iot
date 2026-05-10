"""Fitness and multiclass metrics for P8-b QGA L2."""

from __future__ import annotations

from typing import Any

import numpy as np


def multiclass_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def macro_metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    eps = 1e-12
    total = float(matrix.sum())
    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    fprs: list[float] = []
    for index in range(matrix.shape[0]):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        fpr = fp / (fp + tn + eps)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr)
        per_class[str(index)] = {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return {
        "accuracy": float(np.trace(matrix) / (total + eps)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "macro_fpr": float(np.mean(fprs)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def compute_l2_fitness(metrics: dict[str, Any], selected_features_count: int, total_features: int, weights: dict[str, float]) -> float:
    feature_ratio = float(selected_features_count) / float(total_features)
    return float(
        float(weights["macro_f1_weight"]) * float(metrics.get("macro_f1", 0.0))
        + float(weights["macro_recall_weight"]) * float(metrics.get("macro_recall", 0.0))
        - float(weights["macro_fpr_penalty"]) * float(metrics.get("macro_fpr", 0.0))
        - float(weights["feature_penalty"]) * feature_ratio
    )
