from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str = "configs/fl_config.yaml") -> Dict[str, Any]:
    full_path = get_project_root() / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")

    with full_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config format: {full_path}")

    return data