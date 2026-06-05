from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"

APP_PATH = DASHBOARD_ROOT / "app.py"
DEMO_TEMPLATE_PATH = DASHBOARD_ROOT / "templates" / "demo.html"
DEMO_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "demo.js"
NOTIFICATIONS_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "notifications.js"
AUDIO_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "audio_signals.js"
CSS_PATH = DASHBOARD_ROOT / "static" / "css" / "dashboard.css"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_dashboard_module():
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    spec = importlib.util.spec_from_file_location("test_p16_18_dashboard_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_p16_18_dashboard_app"] = module
    spec.loader.exec_module(module)
    module.REQUEST_TIMEOUT_SECONDS = 0.05
    module.SERVICE_URLS = {
        "controller": ["http://127.0.0.1:8020"],
        "validator": ["http://127.0.0.1:8015"],
        "bridge": ["http://127.0.0.1:8016"],
        "api": ["http://127.0.0.1:8014"],
    }
    return module


def synthetic_state_with_smartwatch_alerts() -> tuple[dict, dict]:
    node_id = "iot-smart-watch-medium"
    alerts = []
    predictions = []
    windows = []
    flows = []
    for index in range(5):
        flow_id = f"smartwatch-window-{index}"
        severity = "critical" if index < 4 else "high"
        alerts.append(
            {
                "timestamp": f"2026-06-05T14:0{index}:00Z",
                "node_id": node_id,
                "severity": severity,
                "predicted_label": "attack",
                "confidence": 0.8722,
                "flow_id": flow_id,
                "source_topic": f"ids/alerts/{node_id}",
                "received_at_unix": 1000 + index,
            }
        )
        predictions.append(
            {
                "node_id": node_id,
                "flow_id": flow_id,
                "payload": {"predicted_label": "attack", "confidence": 0.8722, "flow_id": flow_id},
                "received_at_unix": 1000 + index,
            }
        )
        windows.append(
            {
                "node_id": node_id,
                "flow_id": flow_id,
                "payload": {
                    "event_type": "smartwatch_packet_window_update",
                    "last_icmp_count": 30,
                    "protocol_focus": "ICMP/TCP/HTTP-like",
                },
                "received_at_unix": 1000 + index,
            }
        )
        flows.append(
            {
                "node_id": node_id,
                "flow_id": flow_id,
                "payload": {"scenario": "smartwatch_passive", "protocol_focus": "ICMP/TCP/HTTP-like"},
                "received_at_unix": 1000 + index,
            }
        )
    state = {
        "recent_alerts": alerts,
        "recent_predictions": predictions,
        "recent_windows": windows,
        "recent_flows": flows,
        "recent_status": [],
        "topic_counts": {
            f"ids/windows/{node_id}": 5,
            f"ids/flows/{node_id}": 5,
            f"ids/alerts/{node_id}": 5,
        },
    }
    metrics = {
        "nodes": {node_id: {"flows": 5, "predictions": 5, "alerts": 5}},
        "errors": {
            "final_ids_api_prediction_errors_total": 0,
            "final_mqtt_bridge_prediction_errors_total": 0,
        },
    }
    return state, metrics


def command_terms() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]


def test_demo_state_contains_incident_correlation_key() -> None:
    from fastapi.testclient import TestClient

    module = load_dashboard_module()
    client = TestClient(module.app)
    response = client.get("/api/live-lab/demo-state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["incident_correlation"]["enabled"] is True
    assert "incidents" in payload["incident_correlation"]


def test_five_smartwatch_alerts_group_into_one_incident() -> None:
    module = load_dashboard_module()
    state, metrics = synthetic_state_with_smartwatch_alerts()
    result = module.build_incident_correlation(state, metrics)
    assert result["enabled"] is True
    assert len(result["incidents"]) == 1
    incident = result["incidents"][0]
    assert incident["incident_id"] == "incident-iot-smart-watch-medium-icmp-flood-like"
    assert incident["alerts_total"] == 5
    assert incident["detection_windows"] == 5
    assert incident["flows"] == 5
    assert incident["predictions_attack"] == 5
    assert incident["severity_counts"]["critical"] == 4
    assert incident["severity_counts"]["high"] == 1
    assert incident["latest_confidence"] == 0.8722


def test_dashboard_references_audio_and_incident_ui() -> None:
    html = read(DEMO_TEMPLATE_PATH)
    js = read(DEMO_JS_PATH)
    audio = read(AUDIO_JS_PATH)
    css = read(CSS_PATH)
    assert "Correlated Security Incident" in html
    assert "demo-enable-sound" in html
    assert "/static/js/audio_signals.js" in html
    assert "renderIncidentCorrelation" in js
    assert "incident_key" in js or "incident.incident_id" in js
    assert "playDeviceConnectedSound" in audio
    assert "playAttackDetectedSound" in audio
    assert "Web Audio" not in audio or "AudioContext" in audio
    assert ".incident-panel" in css


def test_dashboard_js_and_css_files_exist() -> None:
    for path in [DEMO_JS_PATH, NOTIFICATIONS_JS_PATH, AUDIO_JS_PATH, CSS_PATH]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_no_offensive_commands_in_p16_18_files() -> None:
    files = [APP_PATH, DEMO_TEMPLATE_PATH, DEMO_JS_PATH, NOTIFICATIONS_JS_PATH, AUDIO_JS_PATH, CSS_PATH]
    violations = []
    for path in files:
        text = read(path).lower()
        for term in command_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append((path, term))
    assert violations == []
