from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from data_loader import load_evaluations, load_figures, load_registry, load_summary


DASHBOARD_DIR = Path(__file__).resolve().parent
FINAL_DIR = DASHBOARD_DIR.parent
SCRIPTS_DIR = FINAL_DIR / "src" / "scripts"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LIVE_LAB_DASHBOARD_TIMEOUT", "0.8"))

SERVICE_URLS = {
    "controller": [
        os.getenv("LIVE_LAB_CONTROLLER_URL"),
        "http://live-lab-controller:8020",
        "http://127.0.0.1:8020",
    ],
    "validator": [
        os.getenv("ONLINE_VALIDATOR_URL"),
        "http://online-validator:8015",
        "http://127.0.0.1:8015",
    ],
    "bridge": [
        os.getenv("FINAL_MQTT_BRIDGE_URL"),
        "http://final-mqtt-bridge:8016",
        "http://127.0.0.1:8016",
    ],
    "api": [
        os.getenv("FINAL_IDS_API_URL"),
        "http://final-ids-api:8014",
        "http://127.0.0.1:8014",
    ],
}

MODEL_DEFAULTS = {
    "model_id": "p8_fedavg_qga_l1",
    "selected_mask_id": "conservative_seed_42",
    "supported_input_modes": ["selected_12_scaled", "original_28_scaled"],
    "threshold": None,
}

app = FastAPI(title="QI-FL-IDS-IoT Final L1 Dashboard", version="1.5")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")


def service_candidates(name: str) -> list[str]:
    return [url.rstrip("/") for url in SERVICE_URLS[name] if url]


def fetch_first(name: str, path: str, *, as_json: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    for base_url in service_candidates(name):
        url = f"{base_url}{path}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "p16-9-dashboard/1.0"})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                content = response.read().decode("utf-8", errors="replace")
                data: Any = json.loads(content) if as_json and content else content
                return {
                    "ok": True,
                    "service": name,
                    "url": url,
                    "status_code": getattr(response, "status", None),
                    "data": data,
                }
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    urls = service_candidates(name)
    return {"ok": False, "service": name, "url": urls[0] if urls else "", "errors": errors, "data": {}}


def prometheus_metric_sum(text: str, metric_name: str, labels: dict[str, str] | None = None) -> float:
    labels = labels or {}
    total = 0.0
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(metric_name):
            continue
        if labels and not all(f'{key}="{value}"' in line for key, value in labels.items()):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            total += float(parts[1])
            found = True
        except ValueError:
            continue
    return total if found else 0.0


def topic_family_total(topic_counts: dict[str, Any], family: str) -> int:
    prefix = f"ids/{family}/"
    total = 0
    for topic, value in topic_counts.items():
        if str(topic).startswith(prefix):
            try:
                total += int(value)
            except (TypeError, ValueError):
                continue
    return total


def parse_payload_preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def sample_payload(sample: dict[str, Any]) -> dict[str, Any]:
    payload = sample.get("payload")
    if isinstance(payload, dict):
        return payload
    return parse_payload_preview(sample.get("payload_preview"))


def first_present(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_confidence(payload: dict[str, Any]) -> float | None:
    value = first_present(payload, "confidence", "probability_attack", "attack_probability", "score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def node_id_from_topic(topic: str) -> str:
    parts = topic.split("/")
    return parts[2] if len(parts) >= 3 else "unknown"


def merge_nodes(nodes_payload: dict[str, Any], assignments_payload: dict[str, Any]) -> list[dict[str, Any]]:
    assignments = {
        item.get("node_id"): item
        for item in assignments_payload.get("assignments", [])
        if isinstance(item, dict)
    }
    merged: list[dict[str, Any]] = []
    for node in nodes_payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id", "unknown")
        assignment = assignments.get(node_id, {})
        merged.append(
            {
                **node,
                "assigned_tier": assignment.get("assigned_tier", node.get("assigned_tier", "unknown")),
                "model_id": assignment.get("model_id", MODEL_DEFAULTS["model_id"]),
                "selected_mask_id": assignment.get("selected_mask_id", MODEL_DEFAULTS["selected_mask_id"]),
                "supported_input_modes": assignment.get("supported_input_modes", MODEL_DEFAULTS["supported_input_modes"]),
                "mqtt_publish_topic": assignment.get("mqtt_publish_topic", f"ids/flows/{node_id}"),
                "mqtt_prediction_topic": assignment.get("mqtt_prediction_topic", f"ids/predictions/{node_id}"),
                "mqtt_alert_topic": assignment.get("mqtt_alert_topic", f"ids/alerts/{node_id}"),
                "status": "connected",
            }
        )
    return merged


def extract_recent_alerts(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for sample in summary_payload.get("samples", []):
        if not isinstance(sample, dict) or sample.get("family") != "alerts":
            continue
        topic = str(sample.get("topic", ""))
        payload = sample_payload(sample)
        predicted_label = first_present(payload, "predicted_label", "label", "prediction_label", default="unknown")
        alerts.append(
            {
                "timestamp": payload.get("timestamp") or sample.get("timestamp"),
                "node_id": payload.get("node_id") or node_id_from_topic(topic),
                "severity": str(payload.get("severity", "medium")).lower(),
                "predicted_label": predicted_label,
                "predicted_label_id": payload.get("predicted_label_id"),
                "confidence": normalize_confidence(payload),
                "probability_attack": payload.get("probability_attack"),
                "flow_id": payload.get("flow_id") or sample.get("flow_id"),
                "source_topic": topic,
                "received_at_unix": sample.get("received_at_unix"),
                "payload_parse_status": "structured" if "payload" in sample else ("preview_json" if payload else "preview_unavailable"),
            }
        )
    return sorted(alerts, key=lambda item: item.get("received_at_unix") or 0, reverse=True)[:12]


def platform_status(services: dict[str, dict[str, Any]], api_errors: float, bridge_errors: float) -> str:
    if any(not service.get("ok") for service in services.values()):
        return "degraded"
    if api_errors > 0 or bridge_errors > 0:
        return "attention"
    return "operational"


def build_live_lab_state() -> dict[str, Any]:
    from datetime import UTC, datetime

    controller_health = fetch_first("controller", "/health")
    controller_ready = fetch_first("controller", "/ready")
    nodes_response = fetch_first("controller", "/nodes")
    assignments_response = fetch_first("controller", "/assignments")
    validator_health = fetch_first("validator", "/health")
    validator_summary = fetch_first("validator", "/summary")
    bridge_ready = fetch_first("bridge", "/ready")
    bridge_metrics = fetch_first("bridge", "/metrics", as_json=False)
    api_ready = fetch_first("api", "/ready")
    api_metrics = fetch_first("api", "/metrics", as_json=False)

    nodes_payload = nodes_response.get("data") if isinstance(nodes_response.get("data"), dict) else {}
    assignments_payload = assignments_response.get("data") if isinstance(assignments_response.get("data"), dict) else {}
    summary_payload = validator_summary.get("data") if isinstance(validator_summary.get("data"), dict) else {}
    topic_counts = summary_payload.get("topic_counts", {}) if isinstance(summary_payload.get("topic_counts"), dict) else {}
    family_counts = summary_payload.get("family_counts", {}) if isinstance(summary_payload.get("family_counts"), dict) else {}
    bridge_text = bridge_metrics.get("data") if isinstance(bridge_metrics.get("data"), str) else ""
    api_text = api_metrics.get("data") if isinstance(api_metrics.get("data"), str) else ""

    nodes = merge_nodes(nodes_payload, assignments_payload)
    api_errors = prometheus_metric_sum(api_text, "final_ids_api_prediction_errors_total")
    bridge_errors = prometheus_metric_sum(bridge_text, "final_mqtt_bridge_prediction_errors_total")
    flows = int(family_counts.get("flows") or topic_family_total(topic_counts, "flows"))
    predictions = int(family_counts.get("predictions") or topic_family_total(topic_counts, "predictions"))
    alerts = int(family_counts.get("alerts") or topic_family_total(topic_counts, "alerts"))

    services = {
        "controller": {"ok": controller_health["ok"] and controller_ready["ok"], "health": controller_health, "ready": controller_ready},
        "validator": {"ok": validator_health["ok"] and validator_summary["ok"], "health": validator_health, "summary": validator_summary},
        "bridge": {"ok": bridge_ready["ok"] and bridge_metrics["ok"], "ready": bridge_ready, "metrics": {k: v for k, v in bridge_metrics.items() if k != "data"}},
        "api": {"ok": api_ready["ok"] and api_metrics["ok"], "ready": api_ready, "metrics": {k: v for k, v in api_metrics.items() if k != "data"}},
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "platform": {"status": platform_status(services, api_errors, bridge_errors)},
        "kpis": {
            "connected_devices": len(nodes),
            "flows_observed": flows,
            "predictions": predictions,
            "alerts": alerts,
            "api_errors": int(api_errors),
            "bridge_errors": int(bridge_errors),
        },
        "nodes": nodes,
        "assignments": assignments_payload.get("assignments", []),
        "model_profile": MODEL_DEFAULTS,
        "recent_alerts": extract_recent_alerts(summary_payload),
        "services": services,
        "topic_counts": topic_counts,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "summary": load_summary(),
            "registry": load_registry(),
            "evaluations": load_evaluations(),
        },
    )


@app.get("/api/summary")
async def api_summary() -> dict[str, Any]:
    return load_summary()


@app.get("/api/models")
async def api_models() -> dict[str, Any]:
    return load_registry()


@app.get("/api/evaluations")
async def api_evaluations() -> dict[str, Any]:
    return {"models": load_evaluations()}


@app.get("/api/figures")
async def api_figures() -> dict[str, Any]:
    return load_figures()


@app.get("/api/live-lab/state")
async def api_live_lab_state() -> dict[str, Any]:
    return build_live_lab_state()


@app.post("/api/evaluate/{model_id}")
async def api_evaluate(model_id: str) -> dict[str, Any]:
    registry = load_registry()
    if model_id not in {model.get("model_id") for model in registry.get("models", [])}:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    if SCRIPTS_DIR.exists() and str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if str(DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_DIR))

    from evaluation.evaluator import evaluate_models, write_evaluation_outputs

    rows, warnings = evaluate_models(DASHBOARD_DIR / "model_registry.json")
    write_evaluation_outputs(rows, warnings)
    selected = [row for row in rows if row.get("model_id") == model_id]
    return {"model_id": model_id, "evaluation": selected, "warnings": warnings}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "p13-dashboard"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    summary = load_summary()
    registry = load_registry()
    return {
        "ready": bool(summary and registry.get("models")),
        "summary_loaded": bool(summary),
        "models": len(registry.get("models", [])),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8013, reload=False)
