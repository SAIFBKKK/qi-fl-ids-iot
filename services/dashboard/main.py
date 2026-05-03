from __future__ import annotations

import os
import random
import string
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.models import router as models_router
from api.nodes import router as nodes_router
from api.metrics import router as metrics_router
from api.qi import router as qi_router
from api.scenarios import router as scenarios_router
from api.system import router as system_router


FL_SERVER_URL = os.getenv("FL_SERVER_URL", "http://fl-server:8080")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")
MLFLOW_EXPERIMENT_IDS = [
    item.strip()
    for item in os.getenv("MLFLOW_EXPERIMENT_IDS", "").split(",")
    if item.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=10.0)
    app.state.fl_server_url = FL_SERVER_URL.rstrip("/")
    app.state.prometheus_url = PROMETHEUS_URL.rstrip("/")
    app.state.mlflow_url = MLFLOW_URL.rstrip("/")
    yield
    await app.state.http.aclose()


app = FastAPI(title="QI-FL-IDS-IoT Dashboard", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(nodes_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(metrics_router)
app.include_router(qi_router)
app.include_router(scenarios_router)
app.include_router(system_router)

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/tab/iot")


@app.get("/tab/{tab_name}")
async def tab_view(tab_name: str, request: Request):
    valid_tabs = {
        "iot": "tab_iot.html",
        "fl": "tab_fl.html",
        "qi": "tab_qi.html",
        "monitoring": "tab_monitoring.html",
    }
    if tab_name not in valid_tabs:
        raise HTTPException(404, "Unknown tab")
    return templates.TemplateResponse(
        request=request,
        name=valid_tabs[tab_name],
        context={
            "request": request,
            "active_tab": tab_name,
            "fl_server_url": FL_SERVER_URL,
            "prometheus_url": PROMETHEUS_URL,
        },
    )


@app.post("/api/connect")
async def api_connect(request: Request) -> dict:
    profiles = [
        {
            "cpu_cores": 1,
            "ram_mb": 512,
            "device_type": "raspberrypi-zero",
            "battery_powered": True,
            "network_quality": "low",
        },
        {
            "cpu_cores": 2,
            "ram_mb": 1024,
            "device_type": "raspberrypi-3",
            "battery_powered": False,
            "network_quality": "medium",
        },
        {
            "cpu_cores": 4,
            "ram_mb": 2048,
            "device_type": "raspberrypi-4",
            "battery_powered": False,
            "network_quality": "medium",
        },
        {
            "cpu_cores": 8,
            "ram_mb": 4096,
            "device_type": "jetson-nano",
            "battery_powered": False,
            "network_quality": "high",
        },
    ]
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    payload = {"node_id": f"dyn_{suffix}", **random.choice(profiles)}

    try:
        response = await request.app.state.http.post(
            f"{request.app.state.fl_server_url}/nodes/register",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"fl-server unreachable: {exc}") from exc


@app.get("/api/fl/runs")
async def api_fl_runs(request: Request, max_results: int = 20) -> dict:
    """Proxy MLflow runs.search to get recent FL runs."""
    try:
        experiment_ids = list(MLFLOW_EXPERIMENT_IDS)
        if not experiment_ids:
            experiments_response = await request.app.state.http.post(
                f"{request.app.state.mlflow_url}/api/2.0/mlflow/experiments/search",
                json={"max_results": 50},
            )
            experiments_response.raise_for_status()
            experiment_ids = [
                str(experiment.get("experiment_id"))
                for experiment in experiments_response.json().get("experiments", [])
                if experiment.get("experiment_id") is not None
            ]

        if not experiment_ids:
            return {"runs": [], "warning": "no experiments found in MLflow"}

        payload = {
            "experiment_ids": experiment_ids,
            "max_results": max_results,
            "order_by": ["attributes.start_time DESC"],
        }
        response = await request.app.state.http.post(
            f"{request.app.state.mlflow_url}/api/2.0/mlflow/runs/search",
            json=payload,
        )
        response.raise_for_status()
        payload = response.json()
        payload.setdefault("runs", [])
        return payload
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"MLflow unreachable: {exc}") from exc


@app.get("/api/fl/schedule")
async def api_fl_schedule(request: Request) -> dict:
    try:
        response = await request.app.state.http.get(f"{request.app.state.fl_server_url}/schedule")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"fl-server unreachable: {exc}") from exc


@app.post("/api/fl/trigger")
async def api_fl_trigger(request: Request) -> dict:
    try:
        response = await request.app.state.http.post(
            f"{request.app.state.fl_server_url}/training/trigger",
            params={"mode": "mock"},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"fl-server unreachable: {exc}") from exc


@app.get("/api/fl/health")
async def api_fl_health(request: Request) -> dict:
    """Kpi summary: combine /health from fl-server."""
    try:
        response = await request.app.state.http.get(f"{request.app.state.fl_server_url}/health")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"fl-server unreachable: {exc}") from exc


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "dashboard", "version": "1.0"}
