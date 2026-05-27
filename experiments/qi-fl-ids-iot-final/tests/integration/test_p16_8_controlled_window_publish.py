from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
RUN_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_run_controlled_window_publish.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_validate_controlled_window_publish_setup.py"


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def dry_run(node_id: str, input_mode: str) -> dict:
    result = run_python(
        RUN_SCRIPT,
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


def test_controlled_window_publish_dry_run_vm1_returns_12_features() -> None:
    payload = dry_run("iot-rpi-weak", "selected_12_scaled")
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["published"] is False
    assert payload["node_id"] == "iot-rpi-weak"
    assert payload["topic"] == "ids/flows/iot-rpi-weak"
    assert payload["features_count"] == 12
    assert len(payload["payload"]["features"]) == 12


def test_controlled_window_publish_dry_run_vm2_returns_28_features() -> None:
    payload = dry_run("iot-smart-watch-medium", "original_28_scaled")
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["published"] is False
    assert payload["node_id"] == "iot-smart-watch-medium"
    assert payload["topic"] == "ids/flows/iot-smart-watch-medium"
    assert payload["features_count"] == 28
    assert len(payload["payload"]["features"]) == 28


def test_validation_setup_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_8_controlled_window_publish_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_p16_8_files_contain_no_command_terms() -> None:
    terms = [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]
    files = [
        RUN_SCRIPT,
        FINAL_ROOT / "src" / "scripts" / "16_8_collect_window_publish_evidence.py",
        VALIDATION_SCRIPT,
        FINAL_ROOT / "outputs" / "reports" / "p16_8_controlled_window_publish_report.md",
        LIVE_LAB_ROOT / "docs" / "p16_8_controlled_window_publish.md",
    ]
    for path in files:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in terms:
            assert term not in text, f"{term} found in {path}"
