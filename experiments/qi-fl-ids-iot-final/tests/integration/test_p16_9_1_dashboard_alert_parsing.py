from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
VALIDATOR_ROOT = FINAL_ROOT / "deployment" / "online_validator"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_9_1_validate_dashboard_alert_parsing.py"


def load_dashboard_module():
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    spec = importlib.util.spec_from_file_location("test_p16_9_1_dashboard_app", DASHBOARD_ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_p16_9_1_dashboard_app"] = module
    spec.loader.exec_module(module)
    return module


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_dashboard_extracts_label_and_confidence_from_structured_alert_payload() -> None:
    module = load_dashboard_module()
    alerts = module.extract_recent_alerts(
        {
            "samples": [
                {
                    "topic": "ids/alerts/iot-smart-watch-medium",
                    "family": "alerts",
                    "flow_id": "p16-7-window-demo",
                    "timestamp": "2026-05-27T15:00:00Z",
                    "received_at_unix": 1.0,
                    "payload_preview": '{"truncated":true',
                    "payload": {
                        "node_id": "iot-smart-watch-medium",
                        "timestamp": "2026-05-27T15:00:00Z",
                        "flow_id": "p16-7-window-demo",
                        "predicted_label": "attack",
                        "predicted_label_id": 1,
                        "confidence": 0.99,
                        "severity": "critical",
                    },
                }
            ]
        }
    )
    assert alerts[0]["predicted_label"] == "attack"
    assert alerts[0]["confidence"] == 0.99
    assert alerts[0]["severity"] == "critical"


def test_dashboard_uses_probability_attack_fallback() -> None:
    module = load_dashboard_module()
    alerts = module.extract_recent_alerts(
        {
            "samples": [
                {
                    "topic": "ids/alerts/iot-rpi-weak",
                    "family": "alerts",
                    "flow_id": "flow-fallback",
                    "received_at_unix": 2.0,
                    "payload": {
                        "node_id": "iot-rpi-weak",
                        "label": "attack",
                        "probability_attack": 0.875,
                        "severity": "high",
                    },
                }
            ]
        }
    )
    assert alerts[0]["predicted_label"] == "attack"
    assert alerts[0]["confidence"] == 0.875


def test_online_validator_samples_keep_structured_payload() -> None:
    text = (VALIDATOR_ROOT / "metrics.py").read_text(encoding="utf-8")
    assert 'sample["payload"] = payload' in text


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_9_1_dashboard_alert_parsing_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True
