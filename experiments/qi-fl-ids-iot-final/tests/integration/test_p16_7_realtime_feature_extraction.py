from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REALTIME_ROOT = LIVE_LAB_ROOT / "realtime_agent"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_7_validate_realtime_feature_extraction_setup.py"
CLI_PATH = REALTIME_ROOT / "run_realtime_window_agent.py"

sys.path.insert(0, str(REALTIME_ROOT))

from feature_extractor import extract_28_features_from_window  # noqa: E402
from flow_window import PacketWindow  # noqa: E402
from packet_capture import SyntheticPacketSource  # noqa: E402


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def run_cli(input_mode: str, node_id: str = "iot-drone-sitl") -> dict:
    result = run_python(
        CLI_PATH,
        "--node-id",
        node_id,
        "--input-mode",
        input_mode,
        "--window-size",
        "30",
        "--max-windows",
        "1",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_packet_window_with_30_packets_is_ready() -> None:
    window = PacketWindow(window_size=30)
    for packet in SyntheticPacketSource(packet_count=30).iter_packets():
        window.add_packet(packet)
    assert window.is_ready()
    summary = window.summary()
    assert summary["packet_count"] == 30
    assert summary["ready"] is True


def test_extract_28_features_from_window_returns_28_values() -> None:
    window = PacketWindow(window_size=30)
    for packet in SyntheticPacketSource(packet_count=30).iter_packets():
        window.add_packet(packet)
    extraction = extract_28_features_from_window(window)
    assert len(extraction["feature_names"]) == 28
    assert len(extraction["features_28"]) == 28
    assert extraction["unsupported_features"] == []
    assert len(extraction["approximated_features"]) == 28


def test_selected_12_scaled_dry_run_returns_12_features() -> None:
    payload = run_cli("selected_12_scaled", "iot-drone-sitl")
    window = payload["windows"][0]
    mqtt_payload = window["publish_result"]["payload"]
    assert payload["dry_run"] is True
    assert window["feature_count"] == 12
    assert len(mqtt_payload["features"]) == 12
    assert mqtt_payload["node_id"] == "iot-drone-sitl"


def test_original_28_unscaled_dry_run_returns_28_features() -> None:
    payload = run_cli("original_28_unscaled", "iot-smart-watch-medium")
    window = payload["windows"][0]
    mqtt_payload = window["publish_result"]["payload"]
    assert window["feature_count"] == 28
    assert len(mqtt_payload["features"]) == 28
    assert mqtt_payload["input_mode"] == "original_28_unscaled"


def test_run_realtime_window_agent_dry_run_works() -> None:
    payload = run_cli("original_28_scaled", "iot-smart-watch-medium")
    assert payload["ok"] is True
    assert payload["source"]["source"] == "synthetic"
    assert payload["windows"][0]["feature_count"] == 28


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_7_realtime_feature_extraction_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_p16_7_files_contain_no_command_terms() -> None:
    terms = [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]
    files = list(REALTIME_ROOT.glob("*.py")) + [
        REALTIME_ROOT / "README.md",
        LIVE_LAB_ROOT / "docs" / "p16_7_realtime_feature_extraction.md",
        FINAL_ROOT / "outputs" / "reports" / "p16_7_realtime_feature_extraction_plan.md",
        VALIDATION_SCRIPT,
    ]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in terms:
            assert term not in text, f"{term} found in {path}"

