from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision_attack = tp / max(tp + fp, 1)
    recall_attack = tp / max(tp + fn, 1)
    f1_attack = 2 * precision_attack * recall_attack / max(precision_attack + recall_attack, 1e-12)
    precision_normal = tn / max(tn + fn, 1)
    recall_normal = tn / max(tn + fp, 1)
    f1_normal = 2 * precision_normal * recall_normal / max(precision_normal + recall_normal, 1e-12)
    support_normal = int((y_true == 0).sum())
    support_attack = int((y_true == 1).sum())
    total = max(int(y_true.size), 1)
    return {
        "accuracy": float((tp + tn) / total),
        "macro_f1": float((f1_normal + f1_attack) / 2.0),
        "weighted_f1": float((support_normal * f1_normal + support_attack * f1_attack) / total),
        "attack_recall": float(recall_attack),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
