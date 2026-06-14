"""QIFA reporting helpers."""

from __future__ import annotations

from typing import Any

from qga.fedavg_adapter import _comparison_with_p5


def comparison_with_p5(config: dict[str, Any], *, alpha: float, clients: int, test_metrics: dict[str, Any]) -> dict[str, Any]:
    return _comparison_with_p5(config, alpha=alpha, clients=clients, test_metrics=test_metrics)
