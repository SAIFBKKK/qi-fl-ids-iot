from __future__ import annotations


def weighted_average(metrics):
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    metric_keys = metrics[0][1].keys()

    for key in metric_keys:
        aggregated[key] = sum(
            num_examples * client_metrics[key]
            for num_examples, client_metrics in metrics
        ) / total_examples

    return aggregated