from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def generate_run_name(experiment: Mapping[str, Any]) -> str:
    """
    Build a short, human-readable MLflow run name from experiment metadata.

    Pattern: <strategy>-<scenario>-<imbalance>-<timestamp>
    Example: fedprox-normal_noniid-class_weights-20260420-143022
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [
        experiment.get("fl_strategy", "unknown"),
        experiment.get("data_scenario", "unknown"),
        experiment.get("imbalance_strategy", "none"),
        ts,
    ]
    return "-".join(str(p) for p in parts)


def generate_experiment_display_name(experiment: Mapping[str, Any]) -> str:
    """
    MLflow experiment (group) name: one per architecture+strategy combination.

    Example: fl-iot-ids/flat_34/fedprox
    """
    arch = experiment.get("architecture", "unknown")
    strategy = experiment.get("fl_strategy", "unknown")
    return f"fl-iot-ids/{arch}/{strategy}"
