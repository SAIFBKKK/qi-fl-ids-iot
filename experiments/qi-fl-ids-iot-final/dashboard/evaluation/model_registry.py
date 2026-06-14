from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_resolver import resolve


def registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "model_registry.json"


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else registry_path()
    return json.loads(target.read_text(encoding="utf-8"))


def load_reported_rows(summary_csv: str | Path) -> dict[str, dict[str, str]]:
    import csv

    path = resolve(summary_csv)
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    method_map = {
        "P5 FedAvg L1": "p5_fedavg_l1",
        "P8 FedAvg + QGA L1": "p8_fedavg_qga_l1",
        "P9 QIFA L1": "p9_qifa_l1",
        "P9 QIFA + QGA L1": "p9_qifa_qga_l1",
    }
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            model_id = method_map.get(row.get("method", ""))
            if model_id:
                rows[model_id] = row
    return rows
