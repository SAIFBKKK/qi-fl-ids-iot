from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any

from fastapi import Body, FastAPI, Response
from fastapi.responses import JSONResponse

from collector import MQTTEdgeGatewayCollector
from config import settings
from inference_api import EdgeInferenceEngine
from metrics import (
    EDGE_GATEWAY_ALERTS_TOTAL,
    EDGE_GATEWAY_INFERENCE_LATENCY_SECONDS,
    EDGE_GATEWAY_MODEL_READY,
    EDGE_GATEWAY_MQTT_CONNECTED,
    EDGE_GATEWAY_PREDICTIONS_TOTAL,
    EDGE_GATEWAY_REJECTED_TOTAL,
    EDGE_GATEWAY_REQUESTS_TOTAL,
    EDGE_GATEWAY_STATUS,
    prometheus_text,
)
from feature_mapper import features_to_ordered_list, get_canonical_feature_names, map_raw_to_features
from preprocessor import EdgeFeaturePreprocessor
from raw_schema import validate_raw_event

VERSION = "p7.5-local-inference"
MODE = "local_inference"

collector = MQTTEdgeGatewayCollector()
preprocessor: EdgeFeaturePreprocessor | None = None
inference_engine: EdgeInferenceEngine | None = None
startup_errors: list[str] = []
feature_names_ready = False
scaler_ready = False


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


def _model_config_path() -> str:
    if settings.model_config_path:
        return settings.model_config_path
    return str(Path(settings.label_mapping_path).parent / "model_config.json")


def _initialize_bundle_components() -> None:
    global preprocessor, inference_engine, startup_errors, feature_names_ready, scaler_ready

    startup_errors = []
    feature_names_ready = False
    scaler_ready = False
    preprocessor = None
    inference_engine = None
    EDGE_GATEWAY_MODEL_READY.set(0)

    try:
        preprocessor = EdgeFeaturePreprocessor(
            feature_names_path=settings.feature_names_path,
            scaler_path=settings.scaler_path,
        )
        feature_names_ready = True
        scaler_ready = True
    except Exception as exc:  # noqa: BLE001
        startup_errors.append(str(exc))

    try:
        inference_engine = EdgeInferenceEngine(
            model_path=settings.model_path,
            label_mapping_path=settings.label_mapping_path,
            model_config_path=_model_config_path(),
            threshold=settings.inference_threshold,
        )
        EDGE_GATEWAY_MODEL_READY.set(1)
    except Exception as exc:  # noqa: BLE001
        startup_errors.append(str(exc))
        inference_engine = None
        EDGE_GATEWAY_MODEL_READY.set(0)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    EDGE_GATEWAY_STATUS.set(1)
    EDGE_GATEWAY_MQTT_CONNECTED.set(0)
    EDGE_GATEWAY_MODEL_READY.set(0)
    collector.start()
    _initialize_bundle_components()
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
def ready(response: Response) -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    model_configured = _model_configured()
    model_ready = inference_engine is not None
    is_ready = model_ready and preprocessor is not None and feature_names_ready and scaler_ready
    if not is_ready:
        response.status_code = 503
    return {
        "ready": is_ready,
        "service": "edge-ids-gateway",
        "mqtt_configured": _mqtt_configured(),
        "model_configured": model_configured,
        "model_ready": model_ready,
        "scaler_ready": scaler_ready,
        "feature_names_ready": feature_names_ready,
        "mode": MODE,
        "reason": "; ".join(startup_errors) if startup_errors else None,
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


@app.post("/infer/raw")
def infer_raw(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    if preprocessor is None or inference_engine is None or not feature_names_ready or not scaler_ready:
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="model_not_ready").inc()
        return JSONResponse(
            status_code=503,
            content={
                "inferred": False,
                "error": "; ".join(startup_errors) if startup_errors else "model or scaler not ready",
            },
        )

    started_at = time.perf_counter()
    try:
        event = validate_raw_event(payload)
        features = map_raw_to_features(event)
        scaled_features = preprocessor.transform(features)
        prediction = inference_engine.predict(scaled_features)
    except ValueError as exc:
        message = str(exc)
        reason = "raw_schema"
        if "mapped feature" in message or "feature vector" in message or "features must" in message:
            reason = "feature_mapping"
        elif "scaled feature vector" in message:
            reason = "inference"
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason=reason).inc()
        status_code = 400 if reason in {"raw_schema", "feature_mapping"} else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "inferred": False,
                "error": message,
            },
        )
    except Exception as exc:  # noqa: BLE001
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="inference").inc()
        return JSONResponse(
            status_code=503,
            content={
                "inferred": False,
                "error": f"inference failed: {exc}",
            },
        )

    latency_seconds = time.perf_counter() - started_at
    EDGE_GATEWAY_INFERENCE_LATENCY_SECONDS.observe(latency_seconds)
    EDGE_GATEWAY_PREDICTIONS_TOTAL.labels(predicted_label=prediction["predicted_label"]).inc()
    if prediction["is_alert"]:
        EDGE_GATEWAY_ALERTS_TOTAL.labels(severity=prediction["severity"]).inc()

    decision = "block" if prediction["is_alert"] else "allow"
    return JSONResponse(
        status_code=200,
        content={
            "inferred": True,
            "gateway_id": settings.gateway_id,
            "node_group": settings.node_group,
            "flow": {
                "node_id": event["node_id"],
                "scenario": event["scenario"],
                "timestamp": event["timestamp"],
            },
            "prediction": prediction,
            "decision": decision,
            "feature_count": len(features),
            "latency_ms": round(latency_seconds * 1000.0, 3),
        },
    )
