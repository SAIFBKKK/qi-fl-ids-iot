from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DEPLOYMENT = FINAL_ROOT / "deployment"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass


def test_live_lab_controller_importable() -> None:
    module = load_module("p16_live_lab_controller_app", DEPLOYMENT / "live_lab_controller" / "app.py")
    assert module.app.title == "QI-FL-IDS-IoT P16 Live Lab Controller"


def test_tier_assignment_policy() -> None:
    module = load_module("p16_tier_assignment", DEPLOYMENT / "live_lab_controller" / "tier_assignment.py")
    assert module.assign_tier(2, 16) == "weak"
    assert module.assign_tier(8, 4) == "weak"
    assert module.assign_tier(4, 16) == "medium"
    assert module.assign_tier(8, 8) == "medium"
    assert module.assign_tier(8, 16) == "powerful"


def test_node_registration_with_testclient() -> None:
    module = load_module("p16_live_lab_controller_app_registration", DEPLOYMENT / "live_lab_controller" / "app.py")
    client = TestClient(module.app)
    response = client.post(
        "/register-node",
        json={
            "node_id": "iot-drone-sitl",
            "hostname": "iot-drone-sitl-node",
            "cpu_count": 2,
            "ram_gb": 2,
            "device_type": "drone_sitl",
            "mqtt_topic": "ids/flows/iot-drone-sitl",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assigned_tier"] == "weak"
    assert payload["selected_mask_id"] == "conservative_seed_42"
    assert payload["supported_input_modes"] == ["selected_12_scaled", "original_28_scaled"]
    assert payload["mqtt_publish_topic"] == "ids/flows/iot-drone-sitl"


def test_scenario_profiles_do_not_contain_offensive_commands() -> None:
    path = DEPLOYMENT / "live_attack_simulator" / "scenario_profiles.py"
    text = path.read_text(encoding="utf-8").lower()
    forbidden = ["h" + "ping3", "hy" + "dra", "ett" + "ercap", "n" + "map", "mass" + "can"]
    assert not [term for term in forbidden if term in text]


def test_agent_builds_12_and_28_feature_payloads() -> None:
    module = load_module("p16_flow_replay", DEPLOYMENT / "live_iot_node_agent" / "flow_replay.py")
    payload_12 = module.build_flow_payload("node-a", "selected_12_scaled", "benign", 1)
    payload_28 = module.build_flow_payload("node-a", "original_28_scaled", "recon_like", 1)
    assert len(payload_12["features"]) == 12
    assert len(payload_28["features"]) == 28
    assert payload_12["input_mode"] == "selected_12_scaled"
    assert payload_28["input_mode"] == "original_28_scaled"


def test_pcap_to_features_interface_without_real_pcap() -> None:
    module = load_module("p16_pcap_to_features", DEPLOYMENT / "live_feature_extractor" / "pcap_to_features.py")
    result = module.extract_pcap_to_records(None)
    assert result.records == []
    assert result.gaps
    summary = module.schema_summary()
    assert summary["feature_count"] == 28
    assert summary["selected_feature_count"] == 12


def test_compose_contains_live_lab_profile() -> None:
    text = (DEPLOYMENT / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "live-lab-controller:" in text
    assert 'profiles: ["live-lab"]' in text
    assert '"8020:8020"' in text


def test_validation_reports_generable() -> None:
    module = load_module("p16_validation", FINAL_ROOT / "src" / "scripts" / "16_validate_live_lab_setup.py")
    result = module.run_validation(run_compose=False, probe_endpoints=False)
    assert result["required_paths"]["ok"]
    assert result["compose"]["service_declared"]
    assert result["feature_artifacts"]["qga_mask_available"]
    assert result["modes_documented"]["selected_12_scaled"]
    assert result["modes_documented"]["original_28_scaled"]
    assert result["no_offensive_commands"]["ok"]
    assert (FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_validation.json").exists()
    assert (FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_validation.md").exists()


