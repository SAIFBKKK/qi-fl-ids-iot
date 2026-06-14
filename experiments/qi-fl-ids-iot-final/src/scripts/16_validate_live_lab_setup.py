from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"
DEPLOYMENT_DIR = FINAL_ROOT / "deployment"
COMPOSE_PATH = DEPLOYMENT_DIR / "docker-compose.final.yml"


REQUIRED_PATHS = [
    FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_audit.md",
    FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_plan.md",
    DEPLOYMENT_DIR / "live_lab_controller" / "app.py",
    DEPLOYMENT_DIR / "live_iot_node_agent" / "agent.py",
    DEPLOYMENT_DIR / "live_attack_simulator" / "scenario_publisher.py",
    DEPLOYMENT_DIR / "live_feature_extractor" / "pcap_to_features.py",
    FINAL_ROOT / "tests" / "integration" / "test_p16_live_lab.py",
]


P16_SCAN_PATHS = [
    DEPLOYMENT_DIR / "live_lab_controller",
    DEPLOYMENT_DIR / "live_iot_node_agent",
    DEPLOYMENT_DIR / "live_attack_simulator",
    DEPLOYMENT_DIR / "live_feature_extractor",
    FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_audit.md",
    FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_plan.md",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def check_required_paths() -> dict[str, Any]:
    missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_compose_config(run_compose: bool = True) -> dict[str, Any]:
    text = read_text(COMPOSE_PATH)
    service_declared = "live-lab-controller:" in text and 'profiles: ["live-lab"]' in text
    result: dict[str, Any] = {"service_declared": service_declared, "config_ok": None, "stdout": "", "stderr": ""}
    if not run_compose:
        result["config_ok"] = "not_run"
        return result
    try:
        completed = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_PATH), "--profile", "live-lab", "config"],
            cwd=DEPLOYMENT_DIR,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["config_ok"] = False
        result["stderr"] = str(exc)
        return result
    result["config_ok"] = completed.returncode == 0
    result["stdout"] = completed.stdout[-2000:]
    result["stderr"] = completed.stderr[-2000:]
    return result


def probe_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read().decode("utf-8")
            return {"reachable": True, "status": response.status, "body": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "status": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:  # noqa: BLE001 - validation must report optional endpoint state.
        return {"reachable": False, "status": None, "error": str(exc)}


def check_endpoints(probe_endpoints: bool = True) -> dict[str, Any]:
    endpoints = {
        "live_lab_controller_health": "http://127.0.0.1:8020/health",
        "final_ids_api_ready": "http://127.0.0.1:8014/ready",
        "final_mqtt_bridge_ready": "http://127.0.0.1:8016/ready",
    }
    if not probe_endpoints:
        return {name: {"reachable": "not_run", "url": url} for name, url in endpoints.items()}
    return {name: {"url": url, **probe_json(url)} for name, url in endpoints.items()}


def check_feature_artifacts() -> dict[str, Any]:
    feature_schema = DEPLOYMENT_DIR / "l1_final" / "feature_schema.json"
    feature_names = FINAL_ROOT / "outputs" / "artifacts" / "features" / "feature_names.json"
    qga_mask = FINAL_ROOT / "outputs" / "qga_feature_selection" / "final_selected_mask" / "feature_mask.json"
    scaler = FINAL_ROOT / "outputs" / "artifacts" / "scalers" / "l1_binary_robust_scaler.pkl"
    schema = json.loads(read_text(feature_schema) or "{}")
    mask = json.loads(read_text(qga_mask) or "{}")
    return {
        "feature_schema_available": feature_schema.exists(),
        "feature_names_available": feature_names.exists(),
        "scaler_available": scaler.exists(),
        "qga_mask_available": qga_mask.exists(),
        "selected_mask_id": mask.get("mask_id"),
        "original_feature_count": schema.get("original_feature_count"),
        "selected_feature_count": schema.get("selected_feature_count"),
    }


def check_modes_documented() -> dict[str, Any]:
    paths = [
        FINAL_ROOT / "outputs" / "reports" / "p16_live_lab_plan.md",
        DEPLOYMENT_DIR / "live_lab_controller" / "README.md",
        DEPLOYMENT_DIR / "live_iot_node_agent" / "README.md",
        DEPLOYMENT_DIR / "live_feature_extractor" / "README.md",
    ]
    combined = "\n".join(read_text(path) for path in paths)
    return {
        "selected_12_scaled": "selected_12_scaled" in combined,
        "original_28_scaled": "original_28_scaled" in combined,
    }


def iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".json"}
    ]


def forbidden_command_terms() -> list[str]:
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


def check_no_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for root in P16_SCAN_PATHS:
        for path in iter_text_files(root):
            text = read_text(path).lower()
            for term in forbidden_command_terms():
                if term in text:
                    violations.append({"file": str(path.relative_to(REPO_ROOT)), "term": term})
    return {"ok": not violations, "violations": violations}


def run_validation(run_compose: bool = True, probe_endpoints: bool = True) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "generated_at": utc_now(),
        "required_paths": check_required_paths(),
        "compose": check_compose_config(run_compose=run_compose),
        "endpoints": check_endpoints(probe_endpoints=probe_endpoints),
        "feature_artifacts": check_feature_artifacts(),
        "modes_documented": check_modes_documented(),
        "no_offensive_commands": check_no_offensive_commands(),
    }
    results["ok"] = all(
        [
            results["required_paths"]["ok"],
            bool(results["compose"]["service_declared"]),
            results["compose"]["config_ok"] in {True, "not_run"},
            bool(results["feature_artifacts"]["feature_schema_available"]),
            bool(results["feature_artifacts"]["qga_mask_available"]),
            all(results["modes_documented"].values()),
            results["no_offensive_commands"]["ok"],
        ]
    )
    write_reports(results)
    return results


def write_reports(results: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_live_lab_validation.json"
    md_path = REPORT_DIR / "p16_live_lab_validation.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# P16 Live Lab Validation",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Overall status: `{'OK' if results['ok'] else 'FAILED'}`",
        f"- Required files: `{'OK' if results['required_paths']['ok'] else 'MISSING'}`",
        f"- Compose service declared: `{results['compose']['service_declared']}`",
        f"- Compose config: `{results['compose']['config_ok']}`",
        f"- Feature schema available: `{results['feature_artifacts']['feature_schema_available']}`",
        f"- QGA mask available: `{results['feature_artifacts']['qga_mask_available']}`",
        f"- Selected mask id: `{results['feature_artifacts']['selected_mask_id']}`",
        f"- Mode selected_12_scaled documented: `{results['modes_documented']['selected_12_scaled']}`",
        f"- Mode original_28_scaled documented: `{results['modes_documented']['original_28_scaled']}`",
        f"- Offensive command scan: `{'OK' if results['no_offensive_commands']['ok'] else 'FAILED'}`",
        "",
        "## Endpoint Probes",
        "",
    ]
    for name, probe in results["endpoints"].items():
        lines.append(f"- `{name}`: reachable=`{probe.get('reachable')}`, status=`{probe.get('status')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = run_validation()
    print(json.dumps({"ok": results["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

