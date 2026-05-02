from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

router = APIRouter()

QI_DATA_PATH = Path(os.getenv("QI_METRICS_PATH", "/app/data/qi_metrics.yaml"))


def load_qi_metrics() -> dict[str, Any]:
    if not QI_DATA_PATH.exists():
        raise HTTPException(503, f"QI metrics file not found at {QI_DATA_PATH}")
    try:
        with QI_DATA_PATH.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception as exc:
        raise HTTPException(500, f"Failed to load QI metrics: {exc}") from exc


@router.get("/api/qi/overview")
def qi_overview() -> dict[str, Any]:
    """Return all QI metrics and metadata for the comparison tab."""
    return load_qi_metrics()


@router.get("/api/qi/methods")
def qi_methods() -> dict[str, list[dict[str, Any]]]:
    """Return short method metadata for the QI comparison tab."""
    data = load_qi_metrics()
    return {
        "methods": [
            {
                "id": method_id,
                "label": method.get("label"),
                "role": method.get("role"),
                "phase": method.get("phase"),
                "status": method.get("status"),
                "color": method.get("color"),
            }
            for method_id, method in (data.get("methods") or {}).items()
        ]
    }


@router.get("/api/qi/metrics_order")
def qi_metrics_order() -> dict[str, list[dict[str, Any]]]:
    """Return metrics order for table rows and radar axes."""
    return {"metrics_order": load_qi_metrics().get("metrics_order", [])}
