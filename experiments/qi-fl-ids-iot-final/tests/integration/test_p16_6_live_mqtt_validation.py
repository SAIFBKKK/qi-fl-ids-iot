from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
PUBLISH_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_6_publish_safe_mqtt_payloads.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_6_validate_live_mqtt_setup.py"

P16_6_FILES = [
    PUBLISH_SCRIPT,
    FINAL_ROOT / "src" / "scripts" / "16_6_collect_live_mqtt_evidence.py",
    VALIDATION_SCRIPT,
    FINAL_ROOT / "outputs" / "reports" / "p16_6_live_mqtt_validation_evidence.md",
    LIVE_LAB_ROOT / "docs" / "p16_6_live_mqtt_screenshot_checklist.md",
    LIVE_LAB_ROOT / "docs" / "p16_6_live_mqtt_node_to_ids_validation.md",
]


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def dry_run_payload(node_id: str, input_mode: str) -> dict:
    result = run_python(PUBLISH_SCRIPT, "--node-id", node_id, "--input-mode", input_mode, "--dry-run")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_publish_safe_payload_dry_run_vm1_has_12_features() -> None:
    output = dry_run_payload("iot-drone-sitl", "selected_12_scaled")
    payload = output["payload"]
    assert output["topic"] == "ids/flows/iot-drone-sitl"
    assert payload["node_id"] == "iot-drone-sitl"
    assert payload["input_mode"] == "selected_12_scaled"
    assert len(payload["features"]) == 12
    assert payload["metadata"]["selected_mask_id"] == "conservative_seed_42"


def test_publish_safe_payload_dry_run_vm2_has_28_features() -> None:
    output = dry_run_payload("iot-smart-watch-medium", "original_28_scaled")
    payload = output["payload"]
    assert output["topic"] == "ids/flows/iot-smart-watch-medium"
    assert payload["node_id"] == "iot-smart-watch-medium"
    assert payload["input_mode"] == "original_28_scaled"
    assert len(payload["features"]) == 28
    assert payload["metadata"]["selected_mask_id"] == "conservative_seed_42"


def test_validation_setup_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_6_live_mqtt_setup_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_p16_6_files_contain_no_offensive_commands() -> None:
    terms = [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]
    for path in P16_6_FILES:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in terms:
            assert term not in text, f"{term} found in {path}"

