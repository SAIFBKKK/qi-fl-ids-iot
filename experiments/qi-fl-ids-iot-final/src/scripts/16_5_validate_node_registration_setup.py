from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

CONTROLLER_DIR = FINAL_ROOT / "deployment" / "live_lab_controller"
TIER_ASSIGNMENT_PATH = CONTROLLER_DIR / "tier_assignment.py"
TEST_NODE_REGISTRATION_PATH = LIVE_LAB_ROOT / "scripts" / "test_node_registration.py"
WEAK_RUN_NODE_PATH = LIVE_LAB_ROOT / "nodes" / "iot_rpi_weak" / "run_node.py"
MEDIUM_RUN_NODE_PATH = LIVE_LAB_ROOT / "nodes" / "iot_smart_watch_medium" / "run_node.py"
DOC_PATH = LIVE_LAB_ROOT / "docs" / "p16_5_vm_node_registration.md"
PLAN_PATH = REPORT_DIR / "p16_5_node_registration_plan.md"

P16_5_FILES = [
    TIER_ASSIGNMENT_PATH,
    CONTROLLER_DIR / "node_registry.py",
    TEST_NODE_REGISTRATION_PATH,
    WEAK_RUN_NODE_PATH,
    MEDIUM_RUN_NODE_PATH,
    DOC_PATH,
    PLAN_PATH,
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def load_module(module_name: str, path: Path, extra_paths: list[Path] | None = None):
    for item in [path.parent, *(extra_paths or [])]:
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_required_files() -> dict[str, Any]:
    required = [
        TEST_NODE_REGISTRATION_PATH,
        WEAK_RUN_NODE_PATH,
        MEDIUM_RUN_NODE_PATH,
        DOC_PATH,
        PLAN_PATH,
        TIER_ASSIGNMENT_PATH,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_tier_rules() -> dict[str, Any]:
    module = load_module("p16_5_tier_assignment", TIER_ASSIGNMENT_PATH)
    checks = {
        "raspberry_like_to_weak": module.assign_tier(99, 99, "raspberry_like") == "weak",
        "drone_sitl_to_weak": module.assign_tier(99, 99, "drone_sitl") == "weak",
        "smart_watch_like_to_medium": module.assign_tier(1, 1.5, "smart_watch_like") == "medium",
        "unknown_low_resource_to_weak": module.assign_tier(1, 1.0, "unknown") == "weak",
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_registration_payloads() -> dict[str, Any]:
    common_dir = LIVE_LAB_ROOT / "nodes" / "common"
    weak = load_module("p16_5_weak_run_node", WEAK_RUN_NODE_PATH, [common_dir])
    medium = load_module("p16_5_medium_run_node", MEDIUM_RUN_NODE_PATH, [common_dir, LIVE_LAB_ROOT / "realtime_agent"])
    weak_payload = weak.build_registration_payload(hostname="weak-host")
    medium_payload = medium.build_registration_payload(hostname="medium-host")
    checks = {
        "weak_cpu_count": weak_payload.get("cpu_count") == 1,
        "weak_ram_gb": weak_payload.get("ram_gb") == 1.0,
        "weak_device_type": weak_payload.get("device_type") == "drone_sitl",
        "weak_mqtt_topic": weak_payload.get("mqtt_topic") == "ids/flows/iot-drone-sitl",
        "medium_cpu_count": medium_payload.get("cpu_count") == 1,
        "medium_ram_gb": medium_payload.get("ram_gb") == 1.5,
        "medium_device_type": medium_payload.get("device_type") == "smart_watch_like",
        "medium_mqtt_topic": medium_payload.get("mqtt_topic") == "ids/flows/iot-smart-watch-medium",
    }
    return {"ok": all(checks.values()), "checks": checks, "weak_payload": weak_payload, "medium_payload": medium_payload}


def check_register_node_usage() -> dict[str, Any]:
    files = [TEST_NODE_REGISTRATION_PATH, WEAK_RUN_NODE_PATH, MEDIUM_RUN_NODE_PATH]
    usage = {rel(path): "/register-node" in path.read_text(encoding="utf-8") for path in files}
    return {"ok": all(usage.values()), "usage": usage}


def offensive_terms() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "flood",
        "scan",
        "brute force",
    ]


def check_no_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_5_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in offensive_terms():
            if re.search(rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+)?{re.escape(term)}(\s|$)", text):
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = check_required_files()
    tiers = check_tier_rules()
    payloads = check_registration_payloads()
    endpoint_usage = check_register_node_usage()
    safety = check_no_offensive_commands()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "tier_rules": tiers,
        "payloads": payloads,
        "endpoint_usage": endpoint_usage,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], tiers["ok"], payloads["ok"], endpoint_usage["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_5_node_registration_setup.json"
    md_path = REPORT_DIR / "p16_5_node_registration_setup.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.5 Node Registration Setup Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Tier rules: `{result['tier_rules']['ok']}`",
        f"- Payloads: `{result['payloads']['ok']}`",
        f"- `/register-node` usage: `{result['endpoint_usage']['ok']}`",
        f"- Safety scan: `{result['safety']['ok']}`",
        "",
        "## Expected Tiers",
        "",
        "- `raspberry_like` -> `weak`",
        "- `drone_sitl` -> `weak`",
        "- `smart_watch_like` -> `medium`",
        "- unknown low-resource nodes -> `weak` fallback",
        "",
    ]
    if result["safety"]["violations"]:
        lines.extend(["## Safety Violations", ""])
        lines.extend(f"- `{item['file']}`: `{item['term']}`" for item in result["safety"]["violations"])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


