from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, Response
from fastapi.responses import JSONResponse

from collector import MQTTEdgeGatewayCollector
from config import settings
from metrics import (
    EDGE_GATEWAY_MODEL_READY,
    EDGE_GATEWAY_MQTT_CONNECTED,
    EDGE_GATEWAY_REJECTED_TOTAL,
    EDGE_GATEWAY_REQUESTS_TOTAL,
    EDGE_GATEWAY_STATUS,
    prometheus_text,
)
from feature_mapper import features_to_ordered_list, get_canonical_feature_names, map_raw_to_features
from raw_schema import validate_raw_event

VERSION = "p7.4-feature-mapper"
MODE = "skeleton"

collector = MQTTEdgeGatewayCollector()


def _mqtt_configured() -> bool:
    return bool(settings.mqtt_broker and settings.mqtt_port and settings.raw_input_topic)


def _model_configured() -> bool:
    return all(
        (
            bool(settings.model_path),
            bool(settings.scaler_path),
            bool(settings.feature_names_path),
            bool(settings.label_mapping_path),
        )
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    EDGE_GATEWAY_STATUS.set(1)
    EDGE_GATEWAY_MQTT_CONNECTED.set(0)
    EDGE_GATEWAY_MODEL_READY.set(0)
    collector.start()
    try:
        yield
    finally:
        collector.stop()
        EDGE_GATEWAY_STATUS.set(0)


app = FastAPI(
    title="QI-FL-IDS-IoT edge-ids-gateway",
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    return {
        "service": "edge-ids-gateway",
        "mode": MODE,
        "version": VERSION,
        "gateway_id": settings.gateway_id,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    return {
        "status": "ok",
        "service": "edge-ids-gateway",
        "gateway_id": settings.gateway_id,
        "node_group": settings.node_group,
        "version": VERSION,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    return {
        "ready": True,
        "service": "edge-ids-gateway",
        "mqtt_configured": _mqtt_configured(),
        "model_configured": _model_configured(),
        "mode": MODE,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=prometheus_text(), media_type="text/plain")


@app.post("/validate/raw")
def validate_raw(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    try:
        event = validate_raw_event(payload)
    except ValueError as exc:
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="raw_schema").inc()
        return JSONResponse(
            status_code=400,
            content={
                "valid": False,
                "error": str(exc),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "valid": True,
            "event": event,
        },
    )


@app.post("/map/features")
def map_features(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    try:
        event = validate_raw_event(payload)
        features = map_raw_to_features(event)
        feature_vector = features_to_ordered_list(features)
    except ValueError as exc:
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="feature_mapping").inc()
        return JSONResponse(
            status_code=400,
            content={
                "mapped": False,
                "error": str(exc),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "mapped": True,
            "feature_count": len(features),
            "feature_names": get_canonical_feature_names(),
            "features": features,
            "feature_vector": feature_vector,
        },
    )
