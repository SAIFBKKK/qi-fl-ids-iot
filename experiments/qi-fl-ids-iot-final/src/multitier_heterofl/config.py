"""Configuration helpers for P7 Multi-tier HeteroFL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def normalize_task(task: str) -> str:
    raw = str(task).lower().strip()
    if raw in {"l1", "l1_binary", "binary"}:
        return "l1_binary"
    if raw in {"l2", "l2_family", "family"}:
        return "l2_family"
    raise ValueError(f"P7 supports only l1/l2, got {task!r}")


def task_short(task: str) -> str:
    return "l1" if normalize_task(task) == "l1_binary" else "l2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def tier_mapping_for_k(config: dict[str, Any], clients: int) -> dict[str, str]:
    mapping = config["tier_mapping"][f"k{int(clients)}"]
    return {str(client_id): str(tier) for client_id, tier in mapping.items()}


def list_scenarios(config: dict[str, Any]) -> list[tuple[float, int]]:
    return [(float(alpha), int(k)) for alpha in config["scenario"]["alphas"] for k in config["scenario"]["clients"]]
