from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
SCENARIOS_DIR = Path(os.getenv("SCENARIOS_DIR", "/app/scenarios"))


@router.get("/api/scenarios")
def list_scenarios() -> dict[str, list[dict[str, Any]]]:
    if not SCENARIOS_DIR.exists():
        return {"scenarios": []}

    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scenarios.append(
                {
                    "id": data.get("id", path.stem),
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "duration_seconds": data.get("duration_seconds", 0),
                    "event_count": len(data.get("events", [])),
                    "filename": path.name,
                }
            )
        except Exception as exc:  # noqa: BLE001 - bad demo files should not hide good ones.
            scenarios.append({"id": path.stem, "name": path.stem, "error": str(exc)})
    return {"scenarios": scenarios}


@router.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> dict[str, Any]:
    if not SCENARIOS_DIR.exists():
        raise HTTPException(404, f"scenario {scenario_id} not found")

    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - keep looking for a valid matching file.
            continue
        if data.get("id") == scenario_id or path.stem == scenario_id:
            return data
    raise HTTPException(404, f"scenario {scenario_id} not found")
