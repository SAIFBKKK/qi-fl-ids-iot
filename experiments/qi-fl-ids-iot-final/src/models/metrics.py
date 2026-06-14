"""Binary IDS metrics for P4 centralized L1 evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TP/TN/FP/FN for attack as the positive class."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute binary classification metrics for normal=0 and attack=1."""

    counts = confusion_counts(y_true, y_pred)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    total = tp + tn + fp + fn

    precision_attack = _safe_div(tp, tp + fp)
    recall_attack = _safe_div(tp, tp + fn)
    f1_attack = _safe_div(2 * precision_attack * recall_attack, precision_attack + recall_attack)

    precision_normal = _safe_div(tn, tn + fn)
    recall_normal = _safe_div(tn, tn + fp)
    f1_normal = _safe_div(2 * precision_normal * recall_normal, precision_normal + recall_normal)

    support_normal = tn + fp
    support_attack = tp + fn
    macro_f1 = (f1_normal + f1_attack) / 2.0
    weighted_f1 = _safe_div(
        f1_normal * support_normal + f1_attack * support_attack,
        support_normal + support_attack,
    )

    return {
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision_attack,
        "recall": recall_attack,
        "f1": f1_attack,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_normal": precision_normal,
        "recall_normal": recall_normal,
        "f1_normal": f1_normal,
        "FPR": _safe_div(fp, fp + tn),
        "FNR": _safe_div(fn, fn + tp),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "support_normal": int(support_normal),
        "support_attack": int(support_attack),
        "support_total": int(total),
    }


def predictions_from_threshold(prob_attack: np.ndarray, threshold: float) -> np.ndarray:
    """Convert attack probabilities to binary labels."""

    return (np.asarray(prob_attack) >= threshold).astype(np.int64)


def threshold_sweep(
    y_true: np.ndarray,
    prob_attack: np.ndarray,
    *,
    start: float,
    stop: float,
    step: float,
) -> list[dict[str, Any]]:
    """Evaluate thresholds from start to stop inclusive."""

    thresholds = np.round(np.arange(start, stop + step / 2.0, step), 10)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = predictions_from_threshold(prob_attack, float(threshold))
        row = {"threshold": float(threshold)}
        row.update(binary_metrics(y_true, y_pred))
        rows.append(row)
    return rows


def select_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select primary and secondary validation thresholds."""

    primary = max(rows, key=lambda row: (row["f1_attack"], row["macro_f1"], -row["FPR"]))
    reasonable = [row for row in rows if row["FPR"] <= 0.05]
    secondary_pool = reasonable if reasonable else rows
    secondary = max(
        secondary_pool,
        key=lambda row: (row["recall_attack"], row["f1_attack"], -row["FPR"]),
    )
    return {
        "primary_threshold": primary["threshold"],
        "primary_objective": "f1_attack",
        "primary_validation_metrics": primary,
        "secondary_threshold": secondary["threshold"],
        "secondary_objective": "recall_attack_with_fpr_control",
        "secondary_validation_metrics": secondary,
        "selection_split": "validation",
        "test_used_for_threshold": False,
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Small sklearn-like classification report."""

    metrics = binary_metrics(y_true, y_pred)
    return {
        "normal": {
            "precision": metrics["precision_normal"],
            "recall": metrics["recall_normal"],
            "f1_score": metrics["f1_normal"],
            "support": metrics["support_normal"],
        },
        "attack": {
            "precision": metrics["precision_attack"],
            "recall": metrics["recall_attack"],
            "f1_score": metrics["f1_attack"],
            "support": metrics["support_attack"],
        },
        "accuracy": metrics["accuracy"],
        "macro_avg": {
            "f1_score": metrics["macro_f1"],
        },
        "weighted_avg": {
            "f1_score": metrics["weighted_f1"],
        },
    }


def optional_auc_metrics(y_true: np.ndarray, prob_attack: np.ndarray) -> dict[str, Any]:
    """Compute ROC-AUC and PR-AUC if sklearn is available."""

    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    except Exception as exc:  # pragma: no cover - depends on optional sklearn
        return {
            "roc_auc": None,
            "pr_auc": None,
            "roc_curve": None,
            "pr_curve": None,
            "warning": f"sklearn metrics unavailable: {exc}",
        }

    fpr, tpr, roc_thresholds = roc_curve(y_true, prob_attack)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, prob_attack)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob_attack)),
        "pr_auc": float(average_precision_score(y_true, prob_attack)),
        "roc_curve": {
            "fpr": fpr.astype(float).tolist(),
            "tpr": tpr.astype(float).tolist(),
            "thresholds": roc_thresholds.astype(float).tolist(),
        },
        "pr_curve": {
            "precision": precision.astype(float).tolist(),
            "recall": recall.astype(float).tolist(),
            "thresholds": pr_thresholds.astype(float).tolist(),
        },
        "warning": None,
    }
