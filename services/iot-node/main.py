from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Response
from loguru import logger

from collector import MQTTFlowCollector
from inference_api import InferenceService
from metrics import NodeMetrics
from node_registration import RegistrationState, collect_hardware_profile, register_with_model_server
from preprocessor import FlowPreprocessor


@dataclass(frozen=True)
class Settings:
    node_id: str
    mqtt_broker: str
    mqtt_port: int
    mqtt_username: str | None
    mqtt_password: str | None
    inference_threshold: float
    model_path: str
    scaler_path: str
    label_mapping_path: str
    model_server_url: str | None
    heartbeat_interval_seconds: int
    log_level: str
    log_format: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            node_id=os.getenv("NODE_ID", "node1"),
            mqtt_broker=os.getenv("MQTT_BROKER", "mosquitto"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_username=os.getenv("MQTT_USERNAME", "ids_user"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
            inference_threshold=float(os.getenv("INFERENCE_THRESHOLD", "0.5")),
            model_path=os.getenv("MODEL_PATH", "/artifacts/global_model.pth"),
            scaler_path=os.getenv("SCALER_PATH", "/artifacts/scaler.pkl"),
            label_mapping_path=os.getenv("LABEL_MAPPING_PATH", "/artifacts/label_mapping.json"),
            model_server_url=os.getenv("MODEL_SERVER_URL"),
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_S", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
        )


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        serialize=settings.log_format.lower() == "json",
        backtrace=False,
        diagnose=False,
    )


settings = Settings.from_env()
configure_logging(settings)
metrics = NodeMetrics()
registration_state = RegistrationState()
preprocessor = FlowPreprocessor(
    scaler_path=settings.scaler_path,
    label_mapping_path=settings.label_mapping_path,
)
inference_service = InferenceService(
    model_path=settings.model_path,
    label_mapping_path=settings.label_mapping_path,
)
collector = MQTTFlowCollector(
    node_id=settings.node_id,
    broker=settings.mqtt_broker,
    port=settings.mqtt_port,
    username=settings.mqtt_username,
    password=settings.mqtt_password,
    threshold=settings.inference_threshold,
    preprocessor=preprocessor,
    inference_service=inference_service,
    metrics=metrics,
)
heartbeat_stop = threading.Event()


def _heartbeat_loop() -> None:
    if not settings.model_server_url:
        return

    url = f"{settings.model_server_url.rstrip('/')}/nodes/{settings.node_id}/heartbeat"
    while not heartbeat_stop.is_set():
        try:
            response = requests.post(url, timeout=3)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - heartbeat must not kill inference.
            logger.warning("heartbeat_failed", node_id=settings.node_id, url=url, error=str(exc))
        heartbeat_stop.wait(settings.heartbeat_interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    global registration_state
    metrics.set_node_status(True)
    profile = collect_hardware_profile()
    registration_state = register_with_model_server(settings.model_server_url, profile)
    metrics.set_assigned_tier(registration_state.assigned_tier)
    logger.info(
        "iot_node_startup",
        node_id=settings.node_id,
        mqtt_broker=settings.mqtt_broker,
        mqtt_port=settings.mqtt_port,
        inference_engine=inference_service.engine_type,
        threshold=settings.inference_threshold,
        assigned_tier=registration_state.assigned_tier,
        model_server_url=settings.model_server_url,
    )
    collector.start()
    heartbeat_stop.clear()
    threading.Thread(target=_heartbeat_loop, daemon=True, name="iot-heartbeat").start()
    try:
        yield
    finally:
        heartbeat_stop.set()
        metrics.set_node_status(False)
        collector.stop()
        logger.info("iot_node_shutdown", node_id=settings.node_id)


app = FastAPI(title="QI-FL-IDS-IoT iot-node", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "node_id": settings.node_id,
        "mqtt_connected": metrics.snapshot()["mqtt_connected"],
        "inference_engine": inference_service.engine_type,
        "model_version": "baseline_fedavg_normal_classweights",
        "assigned_tier": registration_state.assigned_tier,
        "model_source": registration_state.model_source,
        "registration_status": registration_state.status,
        "threshold": settings.inference_threshold,
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    mqtt_connected = metrics.snapshot()["mqtt_connected"]
    model_loaded = inference_service is not None
    preprocessor_ready = preprocessor is not None
    is_ready = model_loaded and preprocessor_ready

    if not is_ready:
        response.status_code = 503

    return {
        "status": "ready" if is_ready else "not_ready",
        "service": "iot-node",
        "node_id": settings.node_id,
        "ready": is_ready,
        "model_loaded": model_loaded,
        "preprocessor_ready": preprocessor_ready,
        "mqtt_connected": mqtt_connected,
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(
        content=metrics.prometheus_text(settings.node_id, default_source_topic=collector.flow_topic),
        media_type="text/plain",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
