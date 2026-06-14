"""Configuration helpers for P8-b QGA L2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"QGA L2 config must be a mapping: {path}")
    return payload


def repo_path(config: dict[str, Any], key_path: str | None = None) -> Path:
    root = Path(config.get("project_root", ".")).resolve()
    if key_path is None:
        return root
    value: Any = config
    for part in key_path.split("."):
        value = value[part]
    return (root / str(value)).resolve()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def rel(path: str | Path, root: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"
