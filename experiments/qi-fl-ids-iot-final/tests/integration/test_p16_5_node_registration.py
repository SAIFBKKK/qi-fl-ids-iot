from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
CONTROLLER_DIR = FINAL_ROOT / "deployment" / "live_lab_controller"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_5_validate_node_registration_setup.py"


def load_module(name: str, path: Path, extra_paths: list[Path] | None = None):
    for module_name in ["app", "metrics", "node_registry", "tier_assignment"]:
        sys.modules.pop(module_name, None)
    inserted: list[str] = []
    for item in [path.parent, *(extra_paths or [])]:
        value = str(item)
        sys.path.insert(0, value)
        inserted.append(value)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for value in inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_test_node_registration_dry_run_works() -> None:
    result = run_python(
        LIVE_LAB_ROOT / "scripts" / "test_node_registration.py",
        "--dry-run",
        "--node-id",
        "iot-smart-watch-medium",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["node_id"] == "iot-smart-watch-medium"
    assert payload["cpu_count"] == 1
    assert payload["ram_gb"] == 1.5
    assert payload["mqtt_topic"] == "ids/flows/iot-smart-watch-medium"


def test_weak_run_node_dry_run_contains_registration_payload() -> None:
    result = run_python(LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["node_id"] == "iot-rpi-weak"
    assert payload["registration_payload"]["ram_gb"] == 1.0
    assert payload["registration_payload"]["device_type"] == "raspberry_like"


def test_medium_run_node_dry_run_contains_registration_payload() -> None:
    result = run_python(LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["node_id"] == "iot-smart-watch-medium"
    assert payload["registration_payload"]["ram_gb"] == 1.5
    assert payload["registration_payload"]["device_type"] == "smart_watch_like"


def test_controller_assigns_expected_tiers_and_schema_stays_compatible() -> None:
    controller_app = load_module("p16_5_controller_app", CONTROLLER_DIR / "app.py", [CONTROLLER_DIR])
    client = TestClient(controller_app.app)

    weak_response = client.post(
        "/register-node",
        json={
            "node_id": "iot-rpi-weak",
            "hostname": "weak-host",
            "cpu_count": 1,
            "ram_gb": 1.0,
            "device_type": "raspberry_like",
            "mqtt_topic": "ids/flows/iot-rpi-weak",
        },
    )
    medium_response = client.post(
        "/register-node",
        json={
            "node_id": "iot-smart-watch-medium",
            "hostname": "medium-host",
            "cpu_count": 1,
            "ram_gb": 1.5,
            "device_type": "smart_watch_like",
            "mqtt_topic": "ids/flows/iot-smart-watch-medium",
        },
    )

    assert weak_response.status_code == 200
    assert medium_response.status_code == 200
    weak_payload = weak_response.json()
    medium_payload = medium_response.json()
    assert weak_payload["assigned_tier"] == "weak"
    assert medium_payload["assigned_tier"] == "medium"
    for payload in [weak_payload, medium_payload]:
        assert payload["model_id"] == "p8_fedavg_qga_l1"
        assert payload["selected_mask_id"] == "conservative_seed_42"
        assert payload["supported_input_modes"] == ["selected_12_scaled", "original_28_scaled"]
        assert payload["mqtt_publish_topic"].startswith("ids/flows/")


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_5_node_registration_setup.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

