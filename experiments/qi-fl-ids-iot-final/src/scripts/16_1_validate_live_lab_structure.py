from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"


REQUIRED_REPORTS = [
    REPORT_DIR / "p16_1_live_lab_realtime_audit.md",
    REPORT_DIR / "p16_1_live_lab_realtime_plan.md",
]

REQUIRED_PATHS = [
    LIVE_LAB_ROOT / "README.md",
    LIVE_LAB_ROOT / "configs" / "server.env.example",
    LIVE_LAB_ROOT / "configs" / "node_weak.yaml",
    LIVE_LAB_ROOT / "configs" / "node_medium.yaml",
    LIVE_LAB_ROOT / "configs" / "lab_network.example.yaml",
    LIVE_LAB_ROOT / "nodes" / "common" / "hardware_profiler.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "mqtt_client.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "feature_schema.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "qga_mask.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "utils.py",
    LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py",
    LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "config.yaml",
    LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "README.md",
    LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py",
    LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "config.yaml",
    LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "README.md",
    LIVE_LAB_ROOT / "realtime_agent" / "packet_capture.py",
    LIVE_LAB_ROOT / "realtime_agent" / "flow_window.py",
    LIVE_LAB_ROOT / "realtime_agent" / "feature_extractor.py",
    LIVE_LAB_ROOT / "realtime_agent" / "scaler_runtime.py",
    LIVE_LAB_ROOT / "realtime_agent" / "edge_inference.py",
    LIVE_LAB_ROOT / "realtime_agent" / "README.md",
    LIVE_LAB_ROOT / "scenario_simulator" / "scenario_publisher.py",
    LIVE_LAB_ROOT / "scenario_simulator" / "scenario_profiles.py",
    LIVE_LAB_ROOT / "scenario_simulator" / "README.md",
    LIVE_LAB_ROOT / "scripts" / "setup_linux_vm.sh",
    LIVE_LAB_ROOT / "scripts" / "test_server_connection.py",
    LIVE_LAB_ROOT / "scripts" / "test_mqtt_connection.py",
    LIVE_LAB_ROOT / "scripts" / "test_node_registration.py",
    LIVE_LAB_ROOT / "scripts" / "run_demo_sequence.py",
    LIVE_LAB_ROOT / "docs" / "vm_setup_virtualbox.md",
    LIVE_LAB_ROOT / "docs" / "network_setup.md",
    LIVE_LAB_ROOT / "docs" / "demo_checklist.md",
    LIVE_LAB_ROOT / "docs" / "safety_scope.md",
]

IMPORTABLE_PYTHON = [
    LIVE_LAB_ROOT / "nodes" / "common" / "hardware_profiler.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "mqtt_client.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "feature_schema.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "qga_mask.py",
    LIVE_LAB_ROOT / "nodes" / "common" / "utils.py",
    LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py",
    LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py",
    LIVE_LAB_ROOT / "realtime_agent" / "packet_capture.py",
    LIVE_LAB_ROOT / "realtime_agent" / "flow_window.py",
    LIVE_LAB_ROOT / "realtime_agent" / "feature_extractor.py",
    LIVE_LAB_ROOT / "realtime_agent" / "scaler_runtime.py",
    LIVE_LAB_ROOT / "realtime_agent" / "edge_inference.py",
    LIVE_LAB_ROOT / "scenario_simulator" / "scenario_publisher.py",
    LIVE_LAB_ROOT / "scenario_simulator" / "scenario_profiles.py",
    LIVE_LAB_ROOT / "scripts" / "test_server_connection.py",
    LIVE_LAB_ROOT / "scripts" / "test_mqtt_connection.py",
    LIVE_LAB_ROOT / "scripts" / "test_node_registration.py",
    LIVE_LAB_ROOT / "scripts" / "run_demo_sequence.py",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def check_required_files(paths: list[Path]) -> dict[str, Any]:
    missing = [rel(path) for path in paths if not path.exists()]
    return {"ok": not missing, "missing": missing}


def import_python_file(path: Path) -> tuple[bool, str | None]:
    module_name = "p16_1_" + "_".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    inserted = [str(path.parent)]
    for extra in [
        LIVE_LAB_ROOT / "nodes" / "common",
        LIVE_LAB_ROOT / "realtime_agent",
        LIVE_LAB_ROOT / "scenario_simulator",
    ]:
        inserted.append(str(extra))
    for item in reversed(inserted):
        if item not in sys.path:
            sys.path.insert(0, item)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return False, "unable to create import spec"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return True, None
    except Exception as exc:  # noqa: BLE001 - validation report should capture import failures.
        return False, str(exc)


def check_imports() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for path in IMPORTABLE_PYTHON:
        ok, error = import_python_file(path)
        if not ok:
            failures.append({"file": rel(path), "error": error or "unknown"})
    return {"ok": not failures, "failures": failures}


def forbidden_terms() -> list[str]:
    return [
        "h" + "ping3",
        "hy" + "dra",
        "ett" + "ercap",
        "n" + "map",
        "mass" + "can",
        "msf" + "console",
        "meta" + "sploit",
        "arp" + "spoof",
        "sql" + "map",
    ]


def text_files(root: Path) -> list[Path]:
    suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".example", ".sh"}
    return [path for path in root.rglob("*") if path.is_file() and (path.suffix.lower() in suffixes or path.name.endswith(".env.example"))]


def check_no_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in text_files(LIVE_LAB_ROOT):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden_terms():
            if term in text:
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def check_artifacts() -> dict[str, Any]:
    feature_names = FINAL_ROOT / "outputs" / "artifacts" / "features" / "feature_names.json"
    qga_mask = FINAL_ROOT / "outputs" / "qga_feature_selection" / "final_selected_mask" / "feature_mask.json"
    compose = FINAL_ROOT / "deployment" / "docker-compose.final.yml"
    scaler = FINAL_ROOT / "outputs" / "artifacts" / "scalers" / "l1_binary_robust_scaler.pkl"
    mask_id = None
    if qga_mask.exists():
        mask_id = json.loads(qga_mask.read_text(encoding="utf-8")).get("mask_id")
    return {
        "feature_names_found": feature_names.exists(),
        "qga_mask_found": qga_mask.exists(),
        "selected_mask_id": mask_id,
        "docker_compose_found": compose.exists(),
        "scaler_available": scaler.exists(),
    }


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports = check_required_files(REQUIRED_REPORTS)
    structure = check_required_files(REQUIRED_PATHS)
    imports = check_imports()
    safety = check_no_offensive_commands()
    artifacts = check_artifacts()
    result = {
        "generated_at": utc_now(),
        "reports": reports,
        "structure": structure,
        "imports": imports,
        "safety_scan": safety,
        "artifacts": artifacts,
    }
    result["ok"] = all(
        [
            reports["ok"],
            structure["ok"],
            imports["ok"],
            safety["ok"],
            artifacts["feature_names_found"],
            artifacts["qga_mask_found"],
            artifacts["selected_mask_id"] == "conservative_seed_42",
            artifacts["docker_compose_found"],
        ]
    )
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_1_live_lab_structure_validation.json"
    md_path = REPORT_DIR / "p16_1_live_lab_structure_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.1 Live Lab Structure Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Audit/plan reports present: `{result['reports']['ok']}`",
        f"- Structure present: `{result['structure']['ok']}`",
        f"- Python files importable: `{result['imports']['ok']}`",
        f"- Safety scan: `{'OK' if result['safety_scan']['ok'] else 'FAILED'}`",
        f"- feature_names.json found: `{result['artifacts']['feature_names_found']}`",
        f"- QGA mask found: `{result['artifacts']['qga_mask_found']}`",
        f"- selected_mask_id: `{result['artifacts']['selected_mask_id']}`",
        f"- docker-compose.final.yml found: `{result['artifacts']['docker_compose_found']}`",
        f"- scaler available: `{result['artifacts']['scaler_available']}`",
        "",
    ]
    if result["structure"]["missing"]:
        lines.extend(["## Missing Structure", ""])
        lines.extend(f"- `{item}`" for item in result["structure"]["missing"])
    if result["imports"]["failures"]:
        lines.extend(["## Import Failures", ""])
        lines.extend(f"- `{item['file']}`: {item['error']}" for item in result["imports"]["failures"])
    if result["safety_scan"]["violations"]:
        lines.extend(["## Safety Violations", ""])
        lines.extend(f"- `{item['file']}`: `{item['term']}`" for item in result["safety_scan"]["violations"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

