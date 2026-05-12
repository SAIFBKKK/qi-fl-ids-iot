"""Config helpers for P10 robustness experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def repo_root() -> Path:
    return Path.cwd().resolve()


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def poison_rate_dir(rate: float) -> str:
    return f"rate_{rate:.1f}" if float(rate).is_integer() else f"rate_{rate}"


def clients_dir(poisoned_clients: int | list[int]) -> str:
    if isinstance(poisoned_clients, int):
        return f"clients_{poisoned_clients}"
    return "clients_" + "-".join(str(int(client)) for client in poisoned_clients)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate
