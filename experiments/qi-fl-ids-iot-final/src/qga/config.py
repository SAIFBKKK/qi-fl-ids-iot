"""Configuration helpers for P8 QGA."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def repo_path(config: dict[str, Any], key_path: str | None = None) -> Path:
    root = Path(config.get("project_root", ".")).resolve()
    if key_path is None:
        return root
    value: Any = config
    for part in key_path.split("."):
        value = value[part]
    return (root / str(value)).resolve()


def make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def relative_to_repo(path: str | Path, config: dict[str, Any]) -> str:
    root = repo_path(config)
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def qga_params(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    params = dict(config["qga"])
    if mode == "smoke":
        smoke = config.get("smoke", {})
        params["population_size"] = int(smoke.get("population_size", params["population_size"]))
        params["generations"] = int(smoke.get("generations", params["generations"]))
        params["max_samples_for_fitness"] = int(
            smoke.get("max_samples_for_fitness", params["max_samples_for_fitness"])
        )
    params["mode"] = mode
    return params
