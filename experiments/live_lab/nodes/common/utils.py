from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def repo_root_from_live_lab() -> Path:
    return Path(__file__).resolve().parents[4]


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the small flat YAML config files used by the dry-run node scripts."""
    values: dict[str, Any] = {}
    config_path = Path(path)
    if not config_path.exists():
        return values
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def dry_run_message(component: str, action: str) -> dict[str, str]:
    return {"component": component, "action": action, "mode": "dry_run"}

