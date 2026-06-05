"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
Phase 2 Live Lab - 2026-06-05.
Integration tests for the safe MAVLink drone PacketWindow pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_PATH = REPO_ROOT / "experiments" / "live_lab" / "nodes" / "iot_drone_sitl" / "mavlink_passive_agent.py"
REALTIME_DIR = REPO_ROOT / "experiments" / "live_lab" / "realtime_agent"
MASK_INDICES = [0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]


def load_agent_module():
    sys.path.insert(0, str(REALTIME_DIR))
    spec = importlib.util.spec_from_file_location("mavlink_passive_agent_test", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mavlink_passive_agent_test"] = module
    spec.loader.exec_module(module)
    return module


def run_agent_cli(scenario: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(AGENT_PATH),
            "--scenario",
            scenario,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_normal_sim_dry_run_produces_28_features() -> None:
    payload = run_agent_cli("normal-sim")
    assert payload["ok"] is True
    assert len(payload["features_28"]) == 28
    assert len(payload["feature_names"]) == 28
    assert payload["mqtt"]["payload"]["node_id"] == "iot-drone-sitl"
    assert payload["mqtt"]["payload"]["input_mode"] == "selected_12_scaled"


def test_qga_mask_indices_select_12_scaled_features() -> None:
    module = load_agent_module()
    window = module.window_from_packets(module.synthetic_packets("normal-sim", 30), 30)
    extraction = module.extract_mavlink_features_28(window)
    transform = module.transform_features(extraction["features_28"])
    expected = [transform["scaled_28"][index] for index in MASK_INDICES]
    assert transform["qga_mask"]["selected_indices"] == MASK_INDICES
    assert transform["selected_12"] == expected
    assert len(transform["selected_12"]) == 12


def test_mqtt_payload_is_valid_json_shape() -> None:
    payload = run_agent_cli("normal-sim")
    mqtt_payload = payload["mqtt"]["payload"]
    for key in ["node_id", "timestamp", "features", "input_mode"]:
        assert key in mqtt_payload
    assert mqtt_payload["node_id"] == "iot-drone-sitl"
    assert isinstance(mqtt_payload["features"], list)
    assert len(mqtt_payload["features"]) == 12


def test_burst_sim_has_high_rate_and_low_iat() -> None:
    payload = run_agent_cli("burst-sim")
    features = payload["features_28"]
    assert features[4] > 100
    assert features[26] < 0.01
    assert features[20] == 30
    assert features[27] == 30


def test_normal_sim_has_low_rate_and_nominal_iat() -> None:
    payload = run_agent_cli("normal-sim")
    features = payload["features_28"]
    assert features[4] < 10
    assert features[26] > 0.5
    assert features[20] == 30
    assert features[27] == 30
