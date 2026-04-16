from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_bundle(
    global_cfg_path: str | Path,
    fl_cfg_path: str | Path,
    model_cfg_path: str | Path,
    data_cfg_path: str | Path,
    imbalance_cfg_path: str | Path,
    node_cfg_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(global_cfg_path)
    cfg = deep_merge(cfg, load_yaml(fl_cfg_path))
    cfg = deep_merge(cfg, load_yaml(model_cfg_path))
    cfg = deep_merge(cfg, load_yaml(data_cfg_path))
    cfg = deep_merge(cfg, load_yaml(imbalance_cfg_path))
    if node_cfg_path is not None:
        cfg = deep_merge(cfg, load_yaml(node_cfg_path))
    return cfg