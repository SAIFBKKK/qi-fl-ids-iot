from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from fastapi import APIRouter, Request

router = APIRouter()

SERVICES = {
    "fl-server": os.getenv("FL_SERVER_URL", "http://fl-server:8080") + "/health",
    "mlflow": os.getenv("MLFLOW_URL", "http://mlflow:5000") + "/health",
    "prometheus": os.getenv("PROMETHEUS_URL", "http://prometheus:9090") + "/-/healthy",
    "grafana": os.getenv("GRAFANA_URL", "http://grafana:3000") + "/api/health",
}


async def probe(name: str, url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    try:
        response = await client.get(url, timeout=3.0)
        return {
            "service": name,
            "url": url,
            "status": "up" if response.status_code < 400 else "degraded",
            "http_status": response.status_code,
            "latency_ms": round(response.elapsed.total_seconds() * 1000, 1)
            if response.elapsed
            else None,
        }
    except Exception as exc:  # noqa: BLE001 - health aggregation must degrade cleanly.
        return {"service": name, "url": url, "status": "down", "error": str(exc)[:120]}


@router.get("/api/system/health")
async def system_health(request: Request) -> dict[str, Any]:
    client: httpx.AsyncClient = request.app.state.http
    results = await asyncio.gather(
        *[probe(name, url, client) for name, url in SERVICES.items()]
    )

    ups = sum(1 for result in results if result["status"] == "up")
    total = len(results)
    if ups == total:
        overall = "healthy"
    elif ups >= total - 1:
        overall = "degraded"
    else:
        overall = "down"

    return {
        "overall": overall,
        "ups": ups,
        "total": total,
        "services": results,
    }
