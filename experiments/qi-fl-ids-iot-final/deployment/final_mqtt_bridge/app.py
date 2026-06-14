from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

from bridge import BridgeSettings, FinalMQTTBridge
from metrics import BridgeMetrics


settings = BridgeSettings.from_env()
metrics = BridgeMetrics()
bridge = FinalMQTTBridge(settings, metrics)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    try:
        bridge.start()
    except Exception as exc:  # noqa: BLE001 - expose startup error through health endpoints.
        metrics.mark_error("startup", str(exc))
        metrics.set_ready(False)
    try:
        yield
    finally:
        bridge.stop()


app = FastAPI(title="QI-FL-IDS-IoT Final MQTT Bridge", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    snapshot = metrics.snapshot()
    return {
        "status": "ok",
        "service": "final-mqtt-bridge",
        "mqtt_connected": snapshot["mqtt_connected"],
        "ready": snapshot["ready"],
        "flows_received_total": snapshot["flows_received_total"],
        "predictions_published_total": snapshot["predictions_published_total"],
        "alerts_published_total": snapshot["alerts_published_total"],
        "last_error": snapshot["last_error"],
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    snapshot = metrics.snapshot()
    ready_state = bool(snapshot["ready"] and snapshot["mqtt_connected"])
    if not ready_state:
        response.status_code = 503
    return {
        "ready": ready_state,
        "service": "final-mqtt-bridge",
        "mqtt_host": settings.mqtt_host,
        "mqtt_port": settings.mqtt_port,
        "subscribe_topic": settings.subscribe_topic,
        "final_ids_api_url": settings.final_ids_api_url,
        "mqtt_connected": snapshot["mqtt_connected"],
        "last_error": snapshot["last_error"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.prometheus_text()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8016, reload=False)
