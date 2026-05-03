from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")


def _prometheus_base_url(request: Request) -> str:
    return getattr(request.app.state, "prometheus_url", PROMETHEUS_URL).rstrip("/")


@router.get("/api/prometheus/query")
async def proxy_query(
    request: Request,
    q: str = Query(..., description="PromQL expression"),
    time: str | None = None,
) -> dict[str, Any]:
    client: httpx.AsyncClient = request.app.state.http
    params = {"query": q}
    if time:
        params["time"] = time

    try:
        response = await client.get(
            f"{_prometheus_base_url(request)}/api/v1/query",
            params=params,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"prometheus unreachable: {exc}") from exc


@router.get("/api/prometheus/range")
async def proxy_range(
    request: Request,
    q: str = Query(..., description="PromQL expression"),
    start: str = Query(..., description="RFC3339 or unix timestamp"),
    end: str = Query(..., description="RFC3339 or unix timestamp"),
    step: str = Query("15s", description="step duration"),
) -> dict[str, Any]:
    client: httpx.AsyncClient = request.app.state.http
    params = {"query": q, "start": start, "end": end, "step": step}

    try:
        response = await client.get(
            f"{_prometheus_base_url(request)}/api/v1/query_range",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"prometheus unreachable: {exc}") from exc
