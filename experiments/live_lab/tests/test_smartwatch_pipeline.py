"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
P16.17 - Continuous Smartwatch Traffic Observer tests.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_DIR = REPO_ROOT / "experiments" / "live_lab" / "nodes" / "iot_smart_watch_medium"
AGENT_PATH = NODE_DIR / "smartwatch_passive_agent.py"
REGISTER_PATH = NODE_DIR / "register_smartwatch_node.py"
PROFILE_PATH = NODE_DIR / "node_profile.yaml"
DRONE_AGENT_PATH = REPO_ROOT / "experiments" / "live_lab" / "nodes" / "iot_drone_sitl" / "mavlink_passive_agent.py"
REALTIME_DIR = REPO_ROOT / "experiments" / "live_lab" / "realtime_agent"


def run_python(path: Path, *args: str, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def run_agent_cli(scenario: str, *extra_args: str) -> dict:
    completed = run_python(
        AGENT_PATH,
        "--scenario",
        scenario,
        "--dry-run",
        *extra_args,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def load_agent_module():
    sys.path.insert(0, str(REALTIME_DIR))
    spec = importlib.util.spec_from_file_location("smartwatch_passive_agent_test", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["smartwatch_passive_agent_test"] = module
    spec.loader.exec_module(module)
    return module


def test_smartwatch_profile_exists() -> None:
    text = PROFILE_PATH.read_text(encoding="utf-8")
    assert "node_id: iot-smart-watch-medium" in text
    assert "device_type: smart_watch" in text
    assert "input_mode: original_28_scaled" in text


def test_register_smartwatch_dry_run_payload() -> None:
    completed = run_python(REGISTER_PATH, "--dry-run")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    registration = payload["payload"]
    assert registration["node_id"] == "iot-smart-watch-medium"
    assert registration["device_type"] == "smart_watch"
    assert registration["mqtt_topic"] == "ids/flows/iot-smart-watch-medium"


def test_lowrate_sim_produces_28_scaled_features() -> None:
    payload = run_agent_cli("icmp-lowrate-sim")
    assert payload["ok"] is True
    assert payload["input_mode"] == "original_28_scaled"
    assert payload["features_count"] == 28
    assert len(payload["features_28"]) == 28
    assert len(payload["features_28_scaled"]) == 28
    assert payload["mqtt"]["payload"]["node_id"] == "iot-smart-watch-medium"
    assert len(payload["mqtt"]["payload"]["features"]) == 28


def test_flood_sim_produces_higher_rate_scaled_than_lowrate() -> None:
    lowrate = run_agent_cli("icmp-lowrate-sim")
    accelerated = run_agent_cli("icmp-flood-sim")
    assert accelerated["features_28_scaled"][4] > lowrate["features_28_scaled"][4]
    assert accelerated["features_28"][26] < lowrate["features_28"][26]


def test_continuous_mode_produces_N_windows() -> None:
    payload = run_agent_cli(
        "icmp-lowrate-sim",
        "--continuous",
        "--window-size",
        "30",
        "--window-stride",
        "15",
        "--max-windows",
        "3",
    )
    assert payload["continuous"] is True
    assert payload["windows_published"] == 3
    assert len(payload["windows"]) == 3
    assert [item["packets_received_total"] for item in payload["windows"]] == [30, 45, 60]


def test_status_payload_schema() -> None:
    payload = run_agent_cli("icmp-lowrate-sim", "--continuous", "--max-windows", "1")
    status_payload = payload["status_events"][-1]["payload"]
    for key in [
        "event_type",
        "agent_status",
        "windows_published",
        "packets_received",
        "last_window_id",
        "uptime_seconds",
        "errors",
        "protocol_focus",
        "interface",
    ]:
        assert key in status_payload
    assert status_payload["event_type"] == "smartwatch_observer_status"
    assert status_payload["windows_published"] == 1


def test_windows_payload_schema() -> None:
    payload = run_agent_cli("icmp-flood-sim", "--continuous", "--max-windows", "1")
    window_payload = payload["windows"][-1]["window_update"]["payload"]
    for key in [
        "event_type",
        "window_number",
        "packet_count",
        "buffer_fill",
        "window_size",
        "window_stride",
        "last_rate_scaled",
        "last_iat_scaled",
        "last_icmp_count",
        "last_tcp_count",
        "last_udp_count",
        "last_number",
        "status",
    ]:
        assert key in window_payload
    assert window_payload["event_type"] == "smartwatch_packet_window_update"
    assert window_payload["buffer_fill"] == 30
    assert window_payload["last_icmp_count"] == 30


def test_log_file_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "p1617.jsonl"
    payload = run_agent_cli(
        "icmp-flood-sim",
        "--continuous",
        "--window-stride",
        "15",
        "--max-windows",
        "2",
        "--log-file",
        str(log_file),
    )
    assert payload["windows_published"] == 2
    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        for key in ["ts", "window_number", "flow_id", "rate_scaled", "iat_scaled", "features_28"]:
            assert key in row
        assert row["protocol"] == "ICMP"
        assert len(row["features_28"]) == 28


def test_udp_tcp_syn_remain_zero_for_icmp() -> None:
    payload = run_agent_cli("icmp-flood-sim")
    features = payload["features_28"]
    module = load_agent_module()
    assert features[module.SYN_FLAG_INDEX] == 0
    assert features[module.SYN_COUNT_INDEX] == 0
    assert features[module.TCP_INDEX] == 0
    assert features[module.UDP_INDEX] == 0
    assert features[module.ICMP_INDEX] == 30
    assert features[module.NUMBER_INDEX] == 30


def test_no_drone_files_broken_by_smartwatch_phase() -> None:
    completed = run_python(DRONE_AGENT_PATH, "--scenario", "normal-sim", "--dry-run")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["node_id"] == "iot-drone-sitl"
    assert len(payload["features_12"]) == 12
