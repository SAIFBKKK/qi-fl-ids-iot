from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def generate_run_name(experiment: Mapping[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [
        experiment.get("fl_strategy", "unknown"),
        experiment.get("data_scenario", "unknown"),
        experiment.get("imbalance_strategy", "none"),
        ts,
    ]
    return "-".join(str(part) for part in parts)


def generate_experiment_display_name(experiment: Mapping[str, Any]) -> str:
    architecture = experiment.get("architecture", "unknown")
    strategy = experiment.get("fl_strategy", "unknown")
    return f"fl-iot-ids/{architecture}/{strategy}"
