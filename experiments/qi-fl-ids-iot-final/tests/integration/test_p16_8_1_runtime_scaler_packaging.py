from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REALTIME_ROOT = LIVE_LAB_ROOT / "realtime_agent"
EXPORT_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_1_export_runtime_scaler_json.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_1_validate_runtime_scaler_packaging.py"
SCALER_JSON_PATH = LIVE_LAB_ROOT / "artifacts" / "l1_binary_robust_scaler.json"
RUN_AGENT_PATH = REALTIME_ROOT / "run_realtime_window_agent.py"

sys.path.insert(0, str(REALTIME_ROOT))

from scaler_runtime import apply_qga_mask, load_optional_scaler, scale_28_features  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def test_export_script_importable() -> None:
    module = load_module("p16_8_1_export_scaler", EXPORT_SCRIPT)
    assert hasattr(module, "export_runtime_scaler_json")


def test_scaler_runtime_loads_json_when_present() -> None:
    assert SCALER_JSON_PATH.exists()
    scaler = load_optional_scaler(SCALER_JSON_PATH)
    assert scaler.available is True
    assert scaler.source == "json"
    assert scaler.path == SCALER_JSON_PATH


def test_json_scaler_transform_28_preserves_length() -> None:
    scaler = load_optional_scaler(SCALER_JSON_PATH)
    scaled = scale_28_features([0.0] * 28, scaler=scaler)
    assert len(scaled) == 28


def test_selected_12_scaled_preserves_length() -> None:
    scaler = load_optional_scaler(SCALER_JSON_PATH)
    scaled = scale_28_features([0.0] * 28, scaler=scaler)
    selected = apply_qga_mask(scaled)
    assert len(selected) == 12


def test_run_agent_uses_json_scaler_for_selected_mode() -> None:
    result = run_python(
        RUN_AGENT_PATH,
        "--node-id",
        "iot-drone-sitl",
        "--input-mode",
        "selected_12_scaled",
        "--window-size",
        "30",
        "--max-windows",
        "1",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    transform = payload["windows"][0]["transform"]
    assert payload["windows"][0]["feature_count"] == 12
    assert transform["scaler"]["available"] is True
    assert transform["scaler"]["used"] is True
    assert transform["scaler"]["source"] == "json"


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_8_1_runtime_scaler_packaging_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

