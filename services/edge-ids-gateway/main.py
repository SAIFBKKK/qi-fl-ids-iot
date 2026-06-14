from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from fastapi import Body, FastAPI, Response
from fastapi.responses import JSONResponse

from collector import MQTTEdgeGatewayCollector
from config import settings
from inference_api import EdgeInferenceEngine
from metrics import (
    EDGE_GATEWAY_ALLOWED_TOTAL,
    EDGE_GATEWAY_ALERTS_TOTAL,
    EDGE_GATEWAY_ARTIFACT_MISSING_TOTAL,
    EDGE_GATEWAY_BLOCKED_TOTAL,
    EDGE_GATEWAY_INFERENCE_LATENCY_SECONDS,
    EDGE_GATEWAY_INFERENCE_READY,
    EDGE_GATEWAY_MODEL_READY,
    EDGE_GATEWAY_MQTT_CONNECTED,
    EDGE_GATEWAY_PREDICTIONS_TOTAL,
    EDGE_GATEWAY_READY,
    EDGE_GATEWAY_REJECTED_TOTAL,
    EDGE_GATEWAY_REQUESTS_TOTAL,
    EDGE_GATEWAY_STATUS,
    prometheus_text,
)
from feature_mapper import features_to_ordered_list, get_canonical_feature_names, map_raw_to_features
from preprocessor import EdgeFeaturePreprocessor
from raw_schema import validate_raw_event

VERSION = "p7.7-observability-hardening"
MODE = "local_inference"

collector: MQTTEdgeGatewayCollector | None = None
preprocessor: EdgeFeaturePreprocessor | None = None
inference_engine: EdgeInferenceEngine | None = None
startup_errors: list[str] = []
feature_names_ready = False
scaler_ready = False
started_at = datetime.now(UTC)
recorded_missing_artifacts: set[str] = set()


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


def _artifact_status() -> dict[str, Any]:
    model_config_path = _model_config_path()
    return {
        "model_path": settings.model_path,
        "model_path_exists": Path(settings.model_path).exists(),
        "scaler_path": settings.scaler_path,
        "scaler_path_exists": Path(settings.scaler_path).exists(),
        "feature_names_path": settings.feature_names_path,
        "feature_names_path_exists": Path(settings.feature_names_path).exists(),
        "label_mapping_path": settings.label_mapping_path,
        "label_mapping_path_exists": Path(settings.label_mapping_path).exists(),
        "model_config_path": model_config_path,
        "model_config_path_exists": Path(model_config_path).exists(),
    }


def _topics_status() -> dict[str, str]:
    return {
        "raw_input": settings.raw_input_topic,
        "accepted": settings.accepted_topic,
        "blocked": settings.blocked_topic,
        "predictions": settings.predictions_topic,
        "alerts": settings.alerts_topic,
        "status": settings.status_topic,
    }


def _record_missing_artifacts() -> None:
    artifact_status = _artifact_status()
    missing_pairs = {
        "model": artifact_status["model_path_exists"],
        "scaler": artifact_status["scaler_path_exists"],
        "feature_names": artifact_status["feature_names_path_exists"],
        "label_mapping": artifact_status["label_mapping_path_exists"],
        "model_config": artifact_status["model_config_path_exists"],
    }
    for artifact, exists in missing_pairs.items():
        if not exists and artifact not in recorded_missing_artifacts:
            EDGE_GATEWAY_ARTIFACT_MISSING_TOTAL.labels(artifact=artifact).inc()
            recorded_missing_artifacts.add(artifact)


def _initialize_bundle_components() -> None:
    global preprocessor, inference_engine, startup_errors, feature_names_ready, scaler_ready

    startup_errors = []
    feature_names_ready = False
    scaler_ready = False
    preprocessor = None
    inference_engine = None
    EDGE_GATEWAY_MODEL_READY.set(0)
    EDGE_GATEWAY_INFERENCE_READY.set(0)
    _record_missing_artifacts()

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

    if preprocessor is not None and inference_engine is not None and feature_names_ready and scaler_ready:
        EDGE_GATEWAY_INFERENCE_READY.set(1)
    else:
        EDGE_GATEWAY_INFERENCE_READY.set(0)


def _mqtt_connected() -> bool:
    return collector.mqtt_connected if collector is not None else False


def _ensure_model_ready() -> None:
    if preprocessor is None or inference_engine is None or not feature_names_ready or not scaler_ready:
        raise RuntimeError("; ".join(startup_errors) if startup_errors else "model not ready")


def _classify_pipeline_error(message: str) -> str:
    lowered = message.lower()
    if "model not ready" in lowered or "cannot load" in lowered:
        return "model_not_ready"
    if "mapped feature" in lowered or "feature vector" in lowered or "features must" in lowered:
        return "feature_mapping"
    if "field '" in lowered or "raw event" in lowered or "schema" in lowered or "ip" in lowered:
        return "raw_schema"
    return "inference"


def _readiness_reason() -> str | None:
    artifact_status = _artifact_status()
    artifact_reason_map = [
        ("model_path_exists", "missing_artifact:model"),
        ("scaler_path_exists", "missing_artifact:scaler"),
        ("feature_names_path_exists", "missing_artifact:feature_names"),
        ("label_mapping_path_exists", "missing_artifact:label_mapping"),
        ("model_config_path_exists", "missing_artifact:model_config"),
    ]
    for key, reason in artifact_reason_map:
        if not artifact_status[key]:
            return reason

    if not _model_configured():
        return "model_not_configured"
    if not feature_names_ready:
        return "feature_names_not_ready"
    if not scaler_ready:
        return "scaler_not_ready"
    if inference_engine is None:
        return "model_not_ready"
    if settings.mqtt_enabled and not _mqtt_connected():
        return "mqtt_enabled_but_not_connected"
    if startup_errors:
        return startup_errors[0]
    return None


def get_runtime_status() -> dict[str, Any]:
    artifact_status = _artifact_status()
    mqtt_enabled = settings.mqtt_enabled
    mqtt_configured = _mqtt_configured()
    mqtt_connected = _mqtt_connected()
    inference_ready = preprocessor is not None and inference_engine is not None and feature_names_ready and scaler_ready
    ready = inference_ready and (mqtt_connected if mqtt_enabled else True)
    reason = _readiness_reason()

    status = {
        "service": "edge-ids-gateway",
        "version": VERSION,
        "gateway_id": settings.gateway_id,
        "node_group": settings.node_group,
        "mode": MODE,
        "mqtt_enabled": mqtt_enabled,
        "mqtt_configured": mqtt_configured,
        "mqtt_connected": mqtt_connected,
        "model_configured": _model_configured(),
        "model_ready": inference_engine is not None,
        "scaler_ready": scaler_ready,
        "feature_names_ready": feature_names_ready,
        "inference_ready": inference_ready,
        "ready": ready,
        "reason": reason,
        "topics": _topics_status(),
        "artifact_paths_exist": {
            "model": artifact_status["model_path_exists"],
            "scaler": artifact_status["scaler_path_exists"],
            "feature_names": artifact_status["feature_names_path_exists"],
            "label_mapping": artifact_status["label_mapping_path_exists"],
            "model_config": artifact_status["model_config_path_exists"],
        },
        "uptime_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 3),
    }
    EDGE_GATEWAY_READY.set(1 if ready else 0)
    EDGE_GATEWAY_INFERENCE_READY.set(1 if inference_ready else 0)
    return status


def run_inference_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_model_ready()
    started_at = time.perf_counter()
    event = validate_raw_event(payload)
    features = map_raw_to_features(event)
    scaled_features = preprocessor.transform(features)
    prediction = inference_engine.predict(scaled_features)
    latency_seconds = time.perf_counter() - started_at

    EDGE_GATEWAY_INFERENCE_LATENCY_SECONDS.observe(latency_seconds)
    EDGE_GATEWAY_PREDICTIONS_TOTAL.labels(predicted_label=prediction["predicted_label"]).inc()

    decision = "block" if prediction["is_alert"] else "allow"
    return {
        "event": event,
        "features": features,
        "feature_vector": features_to_ordered_list(features),
        "prediction": prediction,
        "decision": decision,
        "feature_count": len(features),
        "latency_ms": round(latency_seconds * 1000.0, 3),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    global collector, started_at
    started_at = datetime.now(UTC)
    EDGE_GATEWAY_STATUS.set(1)
    EDGE_GATEWAY_READY.set(0)
    EDGE_GATEWAY_MQTT_CONNECTED.set(0)
    EDGE_GATEWAY_MODEL_READY.set(0)
    EDGE_GATEWAY_INFERENCE_READY.set(0)
    _initialize_bundle_components()
    collector = MQTTEdgeGatewayCollector(
        settings=settings,
        preprocessor=preprocessor,
        inference_engine=inference_engine,
        pipeline_runner=run_inference_pipeline,
    )
    if settings.mqtt_enabled:
        try:
            collector.start()
        except Exception as exc:  # noqa: BLE001
            startup_errors.append(f"mqtt startup failed: {exc}")
    try:
        yield
    finally:
        if collector is not None:
            collector.stop()
        EDGE_GATEWAY_STATUS.set(0)
        EDGE_GATEWAY_READY.set(0)


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
    status = get_runtime_status()
    return {
        "status": "ok",
        "service": status["service"],
        "gateway_id": status["gateway_id"],
        "node_group": status["node_group"],
        "version": status["version"],
        "mode": status["mode"],
        "mqtt_enabled": status["mqtt_enabled"],
        "mqtt_connected": status["mqtt_connected"],
        "inference_ready": status["inference_ready"],
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    status = get_runtime_status()
    if not status["ready"]:
        response.status_code = 503
    return status


@app.get("/diagnostics")
def diagnostics() -> dict[str, Any]:
    EDGE_GATEWAY_REQUESTS_TOTAL.inc()
    status = get_runtime_status()
    artifact_status = _artifact_status()
    return {
        "service": status["service"],
        "version": status["version"],
        "gateway_id": status["gateway_id"],
        "node_group": status["node_group"],
        "mode": status["mode"],
        "ready": status["ready"],
        "reason": status["reason"],
        "uptime_seconds": status["uptime_seconds"],
        "mqtt": {
            "enabled": status["mqtt_enabled"],
            "configured": status["mqtt_configured"],
            "connected": status["mqtt_connected"],
            "broker": settings.mqtt_broker,
            "port": settings.mqtt_port,
            "client_id": settings.mqtt_client_id,
            "mqtt_username_set": bool(settings.mqtt_username),
            "mqtt_password_set": bool(settings.mqtt_password),
            "topics": status["topics"],
        },
        "artifacts": {
            "model_path": artifact_status["model_path"],
            "model_path_exists": artifact_status["model_path_exists"],
            "scaler_path": artifact_status["scaler_path"],
            "scaler_path_exists": artifact_status["scaler_path_exists"],
            "feature_names_path": artifact_status["feature_names_path"],
            "feature_names_path_exists": artifact_status["feature_names_path_exists"],
            "label_mapping_path": artifact_status["label_mapping_path"],
            "label_mapping_path_exists": artifact_status["label_mapping_path_exists"],
            "model_config_path": artifact_status["model_config_path"],
            "model_config_path_exists": artifact_status["model_config_path_exists"],
        },
        "inference": {
            "model_ready": status["model_ready"],
            "scaler_ready": status["scaler_ready"],
            "feature_names_ready": status["feature_names_ready"],
            "inference_ready": status["inference_ready"],
            "num_features": len(preprocessor.feature_names) if preprocessor is not None else None,
            "num_classes": inference_engine.num_classes if inference_engine is not None else None,
        },
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
    try:
        result = run_inference_pipeline(payload)
    except ValueError as exc:
        message = str(exc)
        reason = _classify_pipeline_error(message)
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason=reason).inc()
        status_code = 400 if reason in {"raw_schema", "feature_mapping"} else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "inferred": False,
                "error": message,
            },
        )
    except RuntimeError as exc:
        reason = _classify_pipeline_error(str(exc))
        EDGE_GATEWAY_REJECTED_TOTAL.labels(reason=reason).inc()
        return JSONResponse(
            status_code=503,
            content={
                "inferred": False,
                "error": str(exc),
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

    prediction = result["prediction"]
    if result["decision"] == "allow":
        EDGE_GATEWAY_ALLOWED_TOTAL.inc()
    else:
        EDGE_GATEWAY_BLOCKED_TOTAL.inc()
    if prediction["is_alert"]:
        EDGE_GATEWAY_ALERTS_TOTAL.labels(severity=prediction["severity"]).inc()

    return JSONResponse(
        status_code=200,
        content={
            "inferred": True,
            "gateway_id": settings.gateway_id,
            "node_group": settings.node_group,
            "flow": {
                "node_id": result["event"]["node_id"],
                "scenario": result["event"]["scenario"],
                "timestamp": result["event"]["timestamp"],
            },
            "prediction": prediction,
            "decision": result["decision"],
            "feature_count": result["feature_count"],
            "latency_ms": result["latency_ms"],
        },
    )
