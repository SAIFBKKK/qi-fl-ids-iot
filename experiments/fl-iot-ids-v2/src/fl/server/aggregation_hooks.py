from __future__ import annotations

from collections.abc import Iterable
from numbers import Number


FIT_METRIC_KEYS = (
    "train_loss_last",
    "train_time_sec",
    "update_size_bytes",
)

FIT_SUM_KEYS = {
    "train_time_sec",
    "update_size_bytes",
}

EVALUATE_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "recall_macro",
    "benign_recall",
    "false_positive_rate",
    "rare_class_recall",
)


def _weighted_average_by_examples(
    metrics: Iterable[tuple[int, dict]],
    key: str,
) -> float | None:
    total_examples = 0
    weighted_sum = 0.0

    for num_examples, client_metrics in metrics:
        value = client_metrics.get(key)
        if not isinstance(value, Number):
            continue
        total_examples += int(num_examples)
        weighted_sum += float(value) * float(num_examples)

    if total_examples == 0:
        return None

    return float(weighted_sum / total_examples)


def _sum_numeric_metric(metrics: Iterable[tuple[int, dict]], key: str) -> float | None:
    total = 0.0
    found = False
    for _, client_metrics in metrics:
        value = client_metrics.get(key)
        if isinstance(value, Number):
            total += float(value)
            found = True
    return total if found else None


def _aggregate_selected_metrics(
    metrics: list[tuple[int, dict]],
    keys: tuple[str, ...],
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for key in keys:
        value = (
            _sum_numeric_metric(metrics, key)
            if key in FIT_SUM_KEYS
            else _weighted_average_by_examples(metrics, key)
        )
        if value is not None:
            aggregated[key] = value
    return aggregated


def aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict[str, float]:
    return _aggregate_selected_metrics(metrics, FIT_METRIC_KEYS)


def aggregate_evaluate_metrics(metrics: list[tuple[int, dict]]) -> dict[str, float]:
    return _aggregate_selected_metrics(metrics, EVALUATE_METRIC_KEYS)
