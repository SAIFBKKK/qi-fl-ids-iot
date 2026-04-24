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
RARE_CLASS_IDS = (0, 3, 30, 31, 33)


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
    aggregated = _aggregate_selected_metrics(metrics, EVALUATE_METRIC_KEYS)

    count_keys = {
        key
        for _, client_metrics in metrics
        for key in client_metrics
        if key.startswith(("tp_class_", "fp_class_", "fn_class_"))
    }
    summed_counts = {
        key: _sum_numeric_metric(metrics, key)
        for key in sorted(count_keys)
    }
    if summed_counts:
        class_ids = sorted(
            {
                int(key.rsplit("_", 1)[1])
                for key in summed_counts
                if summed_counts[key] is not None
            }
        )
        recalls = []
        f1_scores = []
        for class_id in class_ids:
            tp = float(summed_counts.get(f"tp_class_{class_id}") or 0.0)
            fp = float(summed_counts.get(f"fp_class_{class_id}") or 0.0)
            fn = float(summed_counts.get(f"fn_class_{class_id}") or 0.0)
            support = tp + fn
            if support <= 0 and tp + fp <= 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            recalls.append(recall)
            f1_scores.append(f1)

        rare_recalls = []
        for class_id in RARE_CLASS_IDS:
            tp = float(summed_counts.get(f"tp_class_{class_id}") or 0.0)
            fn = float(summed_counts.get(f"fn_class_{class_id}") or 0.0)
            if tp + fn > 0:
                rare_recalls.append(tp / (tp + fn))

        if recalls:
            aggregated["recall_macro"] = float(sum(recalls) / len(recalls))
        if f1_scores:
            aggregated["macro_f1"] = float(sum(f1_scores) / len(f1_scores))
        if rare_recalls:
            aggregated["rare_class_recall"] = float(sum(rare_recalls) / len(rare_recalls))

    return aggregated
