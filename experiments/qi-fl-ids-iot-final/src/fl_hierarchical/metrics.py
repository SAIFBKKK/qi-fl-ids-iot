"""Multiclass metrics for P6 hierarchical experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[int(truth), int(pred)] += 1
    return matrix


def per_class_from_confusion(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(matrix.sum())
    rows: dict[str, Any] = {}
    f1_values = []
    precision_values = []
    recall_values = []
    weighted_f1 = 0.0
    for class_id, class_name in enumerate(class_names):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        support = int(matrix[class_id, :].sum())
        rows[str(class_id)] = {
            "class_id": int(class_id),
            "class_name": class_name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "FPR": float(fpr),
            "FNR": float(fnr),
        }
        f1_values.append(f1)
        precision_values.append(precision)
        recall_values.append(recall)
        weighted_f1 += f1 * support
    return {
        "per_class": rows,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "precision_macro": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "weighted_f1": float(weighted_f1 / total) if total else 0.0,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "support_total": total,
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    metrics = per_class_from_confusion(matrix, class_names)
    metrics["confusion_matrix"] = matrix.astype(int).tolist()
    return metrics


def classification_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_avg": {
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1_score": metrics["macro_f1"],
        },
        "weighted_avg": {"f1_score": metrics["weighted_f1"]},
        "classes": metrics["per_class"],
    }


def one_vs_rest_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for payload in metrics["per_class"].values():
        rows.append(
            {
                "class_id": int(payload["class_id"]),
                "class_name": payload["class_name"],
                "TP": int(payload["TP"]),
                "FP": int(payload["FP"]),
                "TN": int(payload["TN"]),
                "FN": int(payload["FN"]),
                "precision": float(payload["precision"]),
                "recall": float(payload["recall"]),
                "f1": float(payload["f1"]),
                "FPR": float(payload["FPR"]),
                "FNR": float(payload["FNR"]),
                "support": int(payload["support"]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_confusion_pairs(matrix: list[list[int]], class_names: list[str], *, limit: int = 20) -> list[dict[str, Any]]:
    pairs = []
    arr = np.asarray(matrix, dtype=np.int64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i != j and arr[i, j] > 0:
                pairs.append(
                    {
                        "true_class_id": i,
                        "pred_class_id": j,
                        "true_class": class_names[i],
                        "pred_class": class_names[j],
                        "count": int(arr[i, j]),
                    }
                )
    return sorted(pairs, key=lambda row: row["count"], reverse=True)[:limit]
