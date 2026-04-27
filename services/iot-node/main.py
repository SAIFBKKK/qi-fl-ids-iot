from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from loguru import logger

from collector import MQTTFlowCollector
from inference_api import InferenceService
from metrics import NodeMetrics
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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    metrics.set_node_status(True)
    logger.info(
        "iot_node_startup",
        node_id=settings.node_id,
        mqtt_broker=settings.mqtt_broker,
        mqtt_port=settings.mqtt_port,
        inference_engine=inference_service.engine_type,
        threshold=settings.inference_threshold,
    )
    collector.start()
    try:
        yield
    finally:
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
        "threshold": settings.inference_threshold,
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(
        content=metrics.prometheus_text(settings.node_id, default_source_topic=collector.flow_topic),
        media_type="text/plain",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
