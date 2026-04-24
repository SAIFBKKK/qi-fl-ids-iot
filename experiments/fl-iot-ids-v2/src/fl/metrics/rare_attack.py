from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


DEFAULT_BENIGN_CLASS_ID = 1


def compute_rare_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_class_ids: Iterable[int],
) -> float:
    """
    Mean recall over selected rare classes actually present in y_true.
    """
    rare_class_ids = list(rare_class_ids)
    present_rare = [cls for cls in rare_class_ids if np.any(y_true == cls)]

    if not present_rare:
        return 0.0

    recalls = []
    for cls in present_rare:
        mask = y_true == cls
        denom = int(mask.sum())
        if denom == 0:
            continue
        rec = float((y_pred[mask] == cls).sum() / denom)
        recalls.append(rec)

    return float(np.mean(recalls)) if recalls else 0.0


def compute_benign_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benign_class_id: int = DEFAULT_BENIGN_CLASS_ID,
) -> Dict[str, float]:
    """
    Compute benign recall and false positive rate on benign traffic.

    - benign_recall = TP_benign / actual_benign
    - false_positive_rate = benign predicted as attack? No:
      standard IDS FPR here = malicious prediction among actual benign
      = FP / actual_benign = 1 - benign_recall
    """
    benign_mask = y_true == benign_class_id
    n_benign = int(benign_mask.sum())

    if n_benign == 0:
        return {
            "benign_recall": 0.0,
            "false_positive_rate": 0.0,
        }

    benign_correct = int((y_pred[benign_mask] == benign_class_id).sum())
    benign_recall = benign_correct / n_benign
    false_positive_rate = 1.0 - benign_recall

    return {
        "benign_recall": float(benign_recall),
        "false_positive_rate": float(false_positive_rate),
    }
