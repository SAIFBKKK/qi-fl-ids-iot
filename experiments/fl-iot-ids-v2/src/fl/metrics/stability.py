from __future__ import annotations

from typing import Dict, List


def compute_inter_round_delta(
    previous_value: float | None,
    current_value: float,
) -> float:
    if previous_value is None:
        return 0.0
    return float(current_value - previous_value)


def summarize_series_stability(values: List[float]) -> Dict[str, float]:
    if len(values) <= 1:
        return {
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return {
        "mean_abs_delta": float(sum(deltas) / len(deltas)),
        "max_abs_delta": float(max(deltas)),
    }