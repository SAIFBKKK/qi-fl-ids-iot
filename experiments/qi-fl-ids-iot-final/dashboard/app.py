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

DEMO_NODE_PROFILES = {
    # ancien node: iot-rpi-weak (remplacé phase 2 live lab)
    "iot-drone-sitl": {
        "display_name": "iot-drone-sitl",
        "device_type": "Drone UAV SITL",
        "protocol": "MAVLink/UDP",
        "mavlink_port": 14550,
        "expected_tier": "weak",
        "inference_path": "selected_12_scaled",
        "qga_behavior": "12 selected scaled MAVLink/UDP packet-window features sent directly",
        "description": "Simulated UAV node using passive PacketWindow(30) observation and server-side IDS inference.",
    },
    "iot-smart-watch-medium": {
        "display_name": "iot-smart-watch-medium",
        "device_type": "smart-watch-like",
        "expected_tier": "medium",
        "inference_path": "original_28_scaled",
        "qga_behavior": "QGA mask applied by final-ids-api",
        "description": "Medium IoT node prepared for edge-aware runtime evidence.",
    },
}

DEMO_STEP_TEXT = {
    "platform_ready": "Step 1 - Platform Ready",
    "devices_connected": "Step 2 - Devices Connected",
    "model_assigned": "Step 3 - Model Assigned",
    "packet_window_generated": "Step 4 - Packet Window Generated",
    "mqtt_flows_seen": "Step 5 - MQTT Flow Received",
    "predictions_seen": "Step 6 - IDS Prediction Produced",
    "alerts_seen": "Step 7 - Alert Detected",
    "zero_errors": "Step 8 - Zero Runtime Errors",
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


def metric_int(text: str, metric_name: str, labels: dict[str, str] | None = None) -> int:
    return int(prometheus_metric_sum(text, metric_name, labels))


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


def assignments_by_node(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item.get("node_id"): item
        for item in assignments
        if isinstance(item, dict) and item.get("node_id")
    }


def build_demo_devices(state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = {
        node.get("node_id"): node
        for node in state.get("nodes", [])
        if isinstance(node, dict) and node.get("node_id")
    }
    assignments = assignments_by_node(state.get("assignments", []))
    demo_devices: list[dict[str, Any]] = []

    for node_id, profile in DEMO_NODE_PROFILES.items():
        node = nodes.get(node_id, {})
        assignment = assignments.get(node_id, {})
        connected = bool(node)
        model_id = assignment.get("model_id") or node.get("model_id") or MODEL_DEFAULTS["model_id"]
        selected_mask_id = (
            assignment.get("selected_mask_id")
            or node.get("selected_mask_id")
            or MODEL_DEFAULTS["selected_mask_id"]
        )
        demo_devices.append(
            {
                "node_id": node_id,
                "hostname": node.get("hostname", "waiting"),
                "device_type": node.get("device_type") or profile["device_type"],
                "display_device_type": profile["device_type"],
                "protocol": profile.get("protocol"),
                "mavlink_port": profile.get("mavlink_port"),
                "cpu_count": node.get("cpu_count"),
                "ram_gb": node.get("ram_gb"),
                "assigned_tier": assignment.get("assigned_tier") or node.get("assigned_tier") or profile["expected_tier"],
                "expected_tier": profile["expected_tier"],
                "model_id": model_id,
                "selected_mask_id": selected_mask_id,
                "supported_input_modes": assignment.get("supported_input_modes", MODEL_DEFAULTS["supported_input_modes"]),
                "mqtt_publish_topic": assignment.get("mqtt_publish_topic", f"ids/flows/{node_id}"),
                "mqtt_prediction_topic": assignment.get("mqtt_prediction_topic", f"ids/predictions/{node_id}"),
                "mqtt_alert_topic": assignment.get("mqtt_alert_topic", f"ids/alerts/{node_id}"),
                "registered_at": node.get("registered_at"),
                "updated_at": node.get("updated_at"),
                "status": "connected" if connected else "waiting",
                "connected": connected,
                "inference_path": profile["inference_path"],
                "qga_behavior": profile["qga_behavior"],
                "description": profile["description"],
            }
        )
    return demo_devices


def build_demo_metrics(state: dict[str, Any], bridge_text: str, api_text: str) -> dict[str, Any]:
    node_metrics: dict[str, dict[str, int]] = {}
    for node_id in DEMO_NODE_PROFILES:
        node_metrics[node_id] = {
            "flows": metric_int(bridge_text, "final_mqtt_bridge_flows_received_total", {"node_id": node_id}),
            "predictions": metric_int(
                bridge_text, "final_mqtt_bridge_predictions_published_total", {"node_id": node_id}
            ),
            "alerts": metric_int(bridge_text, "final_mqtt_bridge_alerts_published_total", {"node_id": node_id}),
        }

    api_errors = metric_int(api_text, "final_ids_api_prediction_errors_total")
    bridge_errors = metric_int(bridge_text, "final_mqtt_bridge_prediction_errors_total")
    if api_errors == 0:
        api_errors = int(state.get("kpis", {}).get("api_errors", 0) or 0)
    if bridge_errors == 0:
        bridge_errors = int(state.get("kpis", {}).get("bridge_errors", 0) or 0)

    return {
        "nodes": node_metrics,
        "totals": {
            "flows": sum(item["flows"] for item in node_metrics.values())
            or int(state.get("kpis", {}).get("flows_observed", 0) or 0),
            "predictions": sum(item["predictions"] for item in node_metrics.values())
            or int(state.get("kpis", {}).get("predictions", 0) or 0),
            "alerts": sum(item["alerts"] for item in node_metrics.values())
            or int(state.get("kpis", {}).get("alerts", 0) or 0),
        },
        "errors": {
            "final_ids_api_prediction_errors_total": api_errors,
            "final_mqtt_bridge_prediction_errors_total": bridge_errors,
        },
    }


def build_demo_model_profile(model_info_response: dict[str, Any]) -> dict[str, Any]:
    data = model_info_response.get("data") if isinstance(model_info_response.get("data"), dict) else {}
    threshold = first_present(data, "threshold", "decision_threshold", "alert_threshold", default=0.4)
    return {
        "final_model": "P8 FedAvg + QGA",
        "model_id": first_present(data, "model_id", default=MODEL_DEFAULTS["model_id"]),
        "selected_mask_id": first_present(data, "selected_mask_id", default=MODEL_DEFAULTS["selected_mask_id"]),
        "threshold": threshold,
        "supported_input_modes": first_present(
            data, "supported_input_modes", default=MODEL_DEFAULTS["supported_input_modes"]
        ),
        "vm1_path": "iot-drone-sitl -> PacketWindow(30) -> scaler JSON -> selected_12_scaled -> MQTT -> IDS",
        "vm2_path": "iot-smart-watch-medium -> PacketWindow(30) -> scaler JSON -> original_28_scaled -> API QGA mask -> IDS",
        "scaler": "JSON runtime scaler",
        "qga_behavior": "The weak node may send 12 selected scaled features directly; the medium node may send 28 scaled features and final-ids-api applies the QGA mask.",
        "model_info_available": bool(model_info_response.get("ok")),
    }


def build_demo_steps(
    state: dict[str, Any], devices: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, bool]:
    services = state.get("services", {})
    required_services = ("controller", "validator", "bridge", "api")
    platform_ready = all(bool(services.get(name, {}).get("ok")) for name in required_services)
    devices_connected = all(device.get("connected") for device in devices)
    model_assigned = all(
        device.get("model_id") == MODEL_DEFAULTS["model_id"]
        and device.get("selected_mask_id") == MODEL_DEFAULTS["selected_mask_id"]
        and device.get("assigned_tier") == device.get("expected_tier")
        for device in devices
    )
    totals = metrics.get("totals", {})
    errors = metrics.get("errors", {})
    mqtt_flows_seen = int(totals.get("flows", 0) or 0) > 0
    predictions_seen = int(totals.get("predictions", 0) or 0) > 0
    alerts_seen = int(totals.get("alerts", 0) or 0) > 0 or bool(state.get("recent_alerts"))
    zero_errors = (
        int(errors.get("final_ids_api_prediction_errors_total", 0) or 0) == 0
        and int(errors.get("final_mqtt_bridge_prediction_errors_total", 0) or 0) == 0
    )
    return {
        "platform_ready": platform_ready,
        "devices_connected": devices_connected,
        "model_assigned": model_assigned,
        "packet_window_generated": mqtt_flows_seen,
        "mqtt_flows_seen": mqtt_flows_seen,
        "predictions_seen": predictions_seen,
        "alerts_seen": alerts_seen,
        "zero_errors": zero_errors,
    }


def demo_platform_status(steps: dict[str, bool], state: dict[str, Any]) -> str:
    if all(steps.values()):
        return "ready"
    if any(steps.values()) or any(service.get("ok") for service in state.get("services", {}).values()):
        return "degraded"
    return "offline"


def build_demo_step_details(steps: dict[str, bool]) -> list[dict[str, str]]:
    details = []
    first_pending_seen = False
    explanations = {
        "platform_ready": "Docker services answer health and readiness checks.",
        "devices_connected": "The two VirtualBox IoT nodes are registered in live-lab-controller.",
        "model_assigned": "Each node has the expected tier and final P8 FedAvg + QGA assignment.",
        "packet_window_generated": "A controlled PacketWindow(30) payload has entered the live path.",
        "mqtt_flows_seen": "final-mqtt-bridge observed ids/flows/{node_id}.",
        "predictions_seen": "final-ids-api produced predictions through the MQTT bridge.",
        "alerts_seen": "IDS alerts appeared on ids/alerts/{node_id}.",
        "zero_errors": "Runtime prediction error counters remain at zero.",
    }
    for key, label in DEMO_STEP_TEXT.items():
        done = bool(steps.get(key))
        if done:
            status = "done"
        elif not first_pending_seen:
            status = "active"
            first_pending_seen = True
        else:
            status = "pending"
        details.append({"key": key, "label": label, "status": status, "explanation": explanations[key]})
    return details


def build_demo_events(state: dict[str, Any], devices: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    generated_at = state.get("generated_at")
    events: list[dict[str, Any]] = []
    for device in devices:
        node_id = device["node_id"]
        timestamp = device.get("updated_at") or device.get("registered_at") or generated_at
        if device.get("connected"):
            events.append(
                {
                    "timestamp": timestamp,
                    "type": "device_connected",
                    "title": "device connected",
                    "node_id": node_id,
                    "detail": f"{node_id} registered as {device.get('assigned_tier')} tier.",
                    "severity": "info",
                }
            )
            events.append(
                {
                    "timestamp": timestamp,
                    "type": "model_assigned",
                    "title": "model assigned",
                    "node_id": node_id,
                    "detail": f"{device.get('model_id')} with mask {device.get('selected_mask_id')}.",
                    "severity": "success",
                }
            )
        node_counts = metrics.get("nodes", {}).get(node_id, {})
        if int(node_counts.get("flows", 0) or 0) > 0:
            events.append(
                {
                    "timestamp": generated_at,
                    "type": "flow_published",
                    "title": "flow published",
                    "node_id": node_id,
                    "detail": f"{node_counts.get('flows')} controlled PacketWindow flow(s) observed.",
                    "severity": "info",
                }
            )
        if int(node_counts.get("predictions", 0) or 0) > 0:
            events.append(
                {
                    "timestamp": generated_at,
                    "type": "prediction_received",
                    "title": "prediction received",
                    "node_id": node_id,
                    "detail": f"{node_counts.get('predictions')} IDS prediction(s) published.",
                    "severity": "success",
                }
            )
    for alert in state.get("recent_alerts", [])[:8]:
        events.append(
            {
                "timestamp": alert.get("timestamp") or generated_at,
                "type": "alert_detected",
                "title": "alert detected",
                "node_id": alert.get("node_id"),
                "detail": f"label {alert.get('predicted_label')} confidence {alert.get('confidence')}",
                "severity": alert.get("severity", "medium"),
                "flow_id": alert.get("flow_id"),
            }
        )
    return events[:24]


def build_demo_warnings(
    state: dict[str, Any],
    bridge_metrics: dict[str, Any],
    api_metrics: dict[str, Any],
    model_info: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for name, service in state.get("services", {}).items():
        if not service.get("ok"):
            warnings.append(f"{name} service is not fully reachable from the dashboard container.")
    for response, label in ((bridge_metrics, "final-mqtt-bridge metrics"), (api_metrics, "final-ids-api metrics")):
        if not response.get("ok"):
            warnings.append(f"{label} unavailable: {'; '.join(response.get('errors', [])[:1])}")
    if not model_info.get("ok"):
        warnings.append("final-ids-api /model/info unavailable; dashboard uses the final model defaults.")
    return warnings


def build_demo_state() -> dict[str, Any]:
    state = build_live_lab_state()
    bridge_metrics = fetch_first("bridge", "/metrics", as_json=False)
    api_metrics = fetch_first("api", "/metrics", as_json=False)
    model_info = fetch_first("api", "/model/info")
    bridge_text = bridge_metrics.get("data") if isinstance(bridge_metrics.get("data"), str) else ""
    api_text = api_metrics.get("data") if isinstance(api_metrics.get("data"), str) else ""

    devices = build_demo_devices(state)
    metrics = build_demo_metrics(state, bridge_text, api_text)
    steps = build_demo_steps(state, devices, metrics)
    platform = demo_platform_status(steps, state)
    recent_events = build_demo_events(state, devices, metrics)

    return {
        "generated_at": state.get("generated_at"),
        "platform_status": platform,
        "steps": steps,
        "step_details": build_demo_step_details(steps),
        "devices": devices,
        "assignments": state.get("assignments", []),
        "model_profile": build_demo_model_profile(model_info),
        "latest_alert": state.get("recent_alerts", [None])[0] if state.get("recent_alerts") else None,
        "metrics": metrics,
        "recent_events": recent_events,
        "services": state.get("services", {}),
        "warnings": build_demo_warnings(state, bridge_metrics, api_metrics, model_info),
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


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="demo.html",
        context={"request": request},
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


@app.get("/api/live-lab/demo-state")
async def api_live_lab_demo_state() -> dict[str, Any]:
    return build_demo_state()


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
