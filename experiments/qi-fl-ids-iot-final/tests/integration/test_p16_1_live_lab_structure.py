from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(LIVE_LAB_ROOT / "nodes" / "common"))
    sys.path.insert(0, str(LIVE_LAB_ROOT / "realtime_agent"))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for item in [str(path.parent), str(LIVE_LAB_ROOT / "nodes" / "common"), str(LIVE_LAB_ROOT / "realtime_agent")]:
            try:
                sys.path.remove(item)
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


def test_live_lab_structure_present() -> None:
    assert (LIVE_LAB_ROOT / "README.md").exists()
    assert (LIVE_LAB_ROOT / "configs" / "node_weak.yaml").exists()
    assert (LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py").exists()
    assert (LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py").exists()
    assert (LIVE_LAB_ROOT / "realtime_agent" / "feature_extractor.py").exists()


def test_scripts_importable() -> None:
    common = load_module("p16_1_common_feature_schema", LIVE_LAB_ROOT / "nodes" / "common" / "feature_schema.py")
    qga = load_module("p16_1_common_qga_mask", LIVE_LAB_ROOT / "nodes" / "common" / "qga_mask.py")
    edge = load_module("p16_1_edge_runtime", LIVE_LAB_ROOT / "realtime_agent" / "edge_inference.py")
    assert len(common.expected_28_features()) == 28
    assert qga.selected_mask_id() == "conservative_seed_42"
    assert edge.EdgeInferenceRuntime(enabled=False).describe()["status"] == "placeholder"


def test_dry_run_weak_node_works() -> None:
    result = run_python(LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["node_id"] == "iot-drone-sitl"
    assert payload["input_mode"] == "selected_12_scaled"
    assert len(payload["sample_payload"]["features"]) == 12


def test_dry_run_smart_watch_medium_node_works() -> None:
    result = run_python(LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["node_id"] == "iot-smart-watch-medium"
    assert payload["input_mode"] == "original_28_scaled"
    assert len(payload["sample_payload"]["features"]) == 28
    assert payload["edge_runtime"]["status"] == "placeholder"


def test_validation_script_generates_ok_true() -> None:
    result = run_python(FINAL_ROOT / "src" / "scripts" / "16_1_validate_live_lab_structure.py")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_1_live_lab_structure_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_no_offensive_tools_in_live_lab() -> None:
    forbidden = ["h" + "ping3", "hy" + "dra", "ett" + "ercap", "n" + "map", "mass" + "can"]
    violations: list[tuple[str, str]] = []
    for path in LIVE_LAB_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".sh", ".example"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden:
            if term in text:
                violations.append((str(path), term))
    assert violations == []


