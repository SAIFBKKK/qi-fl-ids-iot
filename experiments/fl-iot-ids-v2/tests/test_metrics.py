import numpy as np

from src.fl.metrics.classification import compute_classification_metrics
from src.fl.metrics.rare_attack import compute_benign_metrics, compute_rare_class_recall
from src.fl.server.aggregation_hooks import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
)


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 2])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "recall_macro" in metrics


def test_benign_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    metrics = compute_benign_metrics(y_true, y_pred, benign_class_id=1)
    assert "benign_recall" in metrics
    assert "false_positive_rate" in metrics


def test_rare_class_recall():
    y_true = np.array([0, 3, 3, 30, 1, 1])
    y_pred = np.array([0, 3, 1, 30, 1, 0])

    value = compute_rare_class_recall(y_true, y_pred, [0, 3, 30, 31, 33])
    assert isinstance(value, float)


def test_aggregate_fit_metrics():
    aggregated = aggregate_fit_metrics(
        [
            (
                100,
                {
                    "train_loss_last": 0.4,
                    "train_time_sec": 10.0,
                    "update_size_bytes": 1000,
                },
            ),
            (
                300,
                {
                    "train_loss_last": 0.2,
                    "train_time_sec": 20.0,
                    "update_size_bytes": 2000,
                },
            ),
        ]
    )

    assert aggregated["train_loss_last"] == 0.25
    assert aggregated["train_time_sec"] == 17.5
    assert aggregated["update_size_bytes"] == 1750.0


def test_aggregate_evaluate_metrics():
    aggregated = aggregate_evaluate_metrics(
        [
            (
                50,
                {
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "recall_macro": 0.3,
                    "benign_recall": 0.8,
                    "false_positive_rate": 0.2,
                    "rare_class_recall": 0.1,
                },
            ),
            (
                150,
                {
                    "accuracy": 0.9,
                    "macro_f1": 0.8,
                    "recall_macro": 0.7,
                    "benign_recall": 0.6,
                    "false_positive_rate": 0.4,
                    "rare_class_recall": 0.5,
                },
            ),
        ]
    )

    assert aggregated["accuracy"] == 0.8
    assert aggregated["macro_f1"] == 0.7
    assert aggregated["recall_macro"] == 0.6
    assert aggregated["benign_recall"] == 0.65
    assert aggregated["false_positive_rate"] == 0.35
    assert aggregated["rare_class_recall"] == 0.4
