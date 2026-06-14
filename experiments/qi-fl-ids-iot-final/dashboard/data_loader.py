from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DASHBOARD_DIR = Path(__file__).resolve().parent
FINAL_DIR = DASHBOARD_DIR.parent
REPORTS_DIR = FINAL_DIR / "outputs" / "reports"
DATA_DIR = DASHBOARD_DIR / "data"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_summary() -> dict[str, Any]:
    return read_json(DATA_DIR / "dashboard_summary.json", {"warnings": ["dashboard_summary_missing"]})


def load_registry() -> dict[str, Any]:
    return read_json(DASHBOARD_DIR / "model_registry.json", {"models": []})


def load_evaluations() -> list[dict[str, str]]:
    return read_csv(REPORTS_DIR / "p13_dashboard_model_evaluation.csv")


def load_figures() -> dict[str, list[dict[str, str]]]:
    return load_summary().get("figures", {})
