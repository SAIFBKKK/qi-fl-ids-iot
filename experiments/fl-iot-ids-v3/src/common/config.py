from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_bundle(
    global_cfg_path: str | Path,
    fl_cfg_path: str | Path,
    model_cfg_path: str | Path,
    data_cfg_path: str | Path,
    imbalance_cfg_path: str | Path,
    node_cfg_path: str | Path | None = None,
) -> Dict[str, Any]:
    cfg = load_yaml(global_cfg_path)
    cfg = deep_merge(cfg, load_yaml(fl_cfg_path))
    cfg = deep_merge(cfg, load_yaml(model_cfg_path))
    cfg = deep_merge(cfg, load_yaml(data_cfg_path))
    cfg = deep_merge(cfg, load_yaml(imbalance_cfg_path))
    if node_cfg_path is not None:
        cfg = deep_merge(cfg, load_yaml(node_cfg_path))
    return cfg


def load_yaml_config(config_path: str = "configs/fl_config.yaml") -> Dict[str, Any]:
    full_path = get_project_root() / config_path
    return load_yaml(full_path)
