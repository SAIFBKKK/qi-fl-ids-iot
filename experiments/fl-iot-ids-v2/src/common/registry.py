from __future__ import annotations

from src.common.config import load_yaml
from src.common.paths import CONFIGS_DIR


def get_experiment_registry() -> dict:
    return load_yaml(CONFIGS_DIR / "experiment_registry.yaml")


def find_experiment(name: str) -> dict:
    registry = get_experiment_registry()
    experiments = registry.get("experiments", [])
    for exp in experiments:
        if exp.get("name") == name:
            return exp
    raise ValueError(f"Experiment not found: {name}")
