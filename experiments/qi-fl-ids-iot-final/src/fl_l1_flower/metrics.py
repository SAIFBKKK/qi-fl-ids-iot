"""Metric aggregation helpers for P5.2 Flower L1."""

from __future__ import annotations

from numbers import Number
from typing import Any

from models.metrics import binary_metrics


def _weighted_average(metrics: list[tuple[int, dict[str, Any]]], key: str) -> float | None:
    total = 0
    weighted = 0.0
    for num_examples, payload in metrics:
        value = payload.get(key)
        if not isinstance(value, Number):
            continue
        total += int(num_examples)
        weighted += float(value) * int(num_examples)
    return float(weighted / total) if total else None


def _sum_metric(metrics: list[tuple[int, dict[str, Any]]], key: str) -> int:
    return int(sum(float(payload.get(key, 0.0)) for _, payload in metrics))


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    """Compute binary metrics from aggregate confusion counts."""

    y_true = [1] * int(tp) + [1] * int(fn) + [0] * int(tn) + [0] * int(fp)
    y_pred = [1] * int(tp) + [0] * int(fn) + [0] * int(tn) + [1] * int(fp)
    import numpy as np

    return binary_metrics(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))


def aggregate_fit_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float | int]:
    """Aggregate client fit metrics for Flower FedAvg."""

    if not metrics:
        return {}
    result: dict[str, float | int] = {}
    for key in ["local_train_loss", "fit_time_sec", "upload_bytes", "download_bytes"]:
        value = _weighted_average(metrics, key)
        if value is not None:
            result[key] = value
    return result


def aggregate_evaluate_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float | int]:
    """Aggregate client validation metrics with confusion-count consistency."""

    if not metrics:
        return {}
    tp = _sum_metric(metrics, "TP")
    tn = _sum_metric(metrics, "TN")
    fp = _sum_metric(metrics, "FP")
    fn = _sum_metric(metrics, "FN")
    aggregated = metrics_from_counts(tp, tn, fp, fn)
    loss = _weighted_average(metrics, "local_val_loss")
    if loss is not None:
        aggregated["loss"] = loss
    return aggregated


def client_metrics_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize Flower client metrics into the P5 CSV schema."""

    return {
        "round": int(metrics.get("round", metrics.get("server_round", 0))),
        "client_id": str(metrics["client_id"]),
        "train_samples": int(metrics.get("train_samples", 0)),
        "val_samples": int(metrics.get("val_samples", 0)),
        "normal_count": int(metrics.get("normal_count", 0)),
        "attack_count": int(metrics.get("attack_count", 0)),
        "local_train_loss": float(metrics.get("local_train_loss", 0.0)),
        "local_val_loss": float(metrics.get("local_val_loss", metrics.get("loss", 0.0))),
        "local_accuracy": float(metrics.get("local_accuracy", metrics.get("accuracy", 0.0))),
        "local_macro_f1": float(metrics.get("local_macro_f1", metrics.get("macro_f1", 0.0))),
        "local_attack_recall": float(metrics.get("local_attack_recall", metrics.get("recall_attack", 0.0))),
        "local_fpr": float(metrics.get("local_fpr", metrics.get("FPR", 0.0))),
        "fit_time_sec": float(metrics.get("fit_time_sec", 0.0)),
        "eval_time_sec": float(metrics.get("eval_time_sec", 0.0)),
        "upload_bytes": int(metrics.get("upload_bytes", 0)),
        "download_bytes": int(metrics.get("download_bytes", 0)),
    }

