from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

from metrics import OnlineValidatorMetrics
from mqtt_observer import MQTTObserver, MQTTSettings


settings = MQTTSettings.from_env()
metrics = OnlineValidatorMetrics()
observer = MQTTObserver(settings, metrics)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    try:
        observer.start()
    except Exception as exc:  # noqa: BLE001 - liveness should expose the startup error.
        metrics.mark_error(str(exc))
    try:
        yield
    finally:
        observer.stop()


app = FastAPI(title="QI-FL-IDS-IoT P15 Online Validator", version="1.7", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    snapshot = metrics.snapshot()
    return {
        "status": "ok",
        "service": "online-validator",
        "mqtt_connected": snapshot["mqtt_connected"],
        "observed_messages": snapshot["total_messages"],
        "last_error": snapshot["last_error"],
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    snapshot = metrics.snapshot()
    ready_state = bool(snapshot["mqtt_connected"])
    if not ready_state:
        response.status_code = 503
    return {
        "ready": ready_state,
        "service": "online-validator",
        "broker": settings.broker,
        "port": settings.port,
        "topics": list(settings.topics),
        "mqtt_connected": snapshot["mqtt_connected"],
        "last_error": snapshot["last_error"],
    }


@app.get("/summary")
def summary() -> dict[str, Any]:
    return metrics.snapshot()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.prometheus_text()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8015, reload=False)
