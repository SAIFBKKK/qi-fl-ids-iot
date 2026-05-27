from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

PUBLISH_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_6_publish_safe_mqtt_payloads.py"
COLLECT_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_6_collect_live_mqtt_evidence.py"
SETUP_VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_6_validate_live_mqtt_setup.py"
EVIDENCE_REPORT = REPORT_DIR / "p16_6_live_mqtt_validation_evidence.md"
SCREENSHOT_DOC = LIVE_LAB_ROOT / "docs" / "p16_6_live_mqtt_screenshot_checklist.md"
VALIDATION_DOC = LIVE_LAB_ROOT / "docs" / "p16_6_live_mqtt_node_to_ids_validation.md"
COMPOSE_PATH = FINAL_ROOT / "deployment" / "docker-compose.final.yml"
FINAL_MQTT_BRIDGE_DIR = FINAL_ROOT / "deployment" / "final_mqtt_bridge"
LIVE_LAB_CONTROLLER_DIR = FINAL_ROOT / "deployment" / "live_lab_controller"

P16_6_FILES = [
    PUBLISH_SCRIPT,
    COLLECT_SCRIPT,
    SETUP_VALIDATION_SCRIPT,
    EVIDENCE_REPORT,
    SCREENSHOT_DOC,
    VALIDATION_DOC,
    REPORT_DIR / "p16_6_live_mqtt_evidence.json",
    REPORT_DIR / "p16_6_live_mqtt_evidence_table.md",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def offensive_terms() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]


def check_required_files() -> dict[str, Any]:
    required = [
        PUBLISH_SCRIPT,
        COLLECT_SCRIPT,
        EVIDENCE_REPORT,
        SCREENSHOT_DOC,
        VALIDATION_DOC,
        COMPOSE_PATH,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_required_services() -> dict[str, Any]:
    checks = {
        "docker_compose_final": COMPOSE_PATH.exists(),
        "final_mqtt_bridge": FINAL_MQTT_BRIDGE_DIR.exists(),
        "live_lab_controller": LIVE_LAB_CONTROLLER_DIR.exists(),
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_script_content() -> dict[str, Any]:
    publish_text = PUBLISH_SCRIPT.read_text(encoding="utf-8") if PUBLISH_SCRIPT.exists() else ""
    collect_text = COLLECT_SCRIPT.read_text(encoding="utf-8") if COLLECT_SCRIPT.exists() else ""
    checks = {
        "publish_uses_ids_flows_topic": "ids/flows/" in publish_text,
        "publish_supports_selected_12_scaled": "selected_12_scaled" in publish_text,
        "publish_supports_original_28_scaled": "original_28_scaled" in publish_text,
        "publish_includes_selected_mask_id": "conservative_seed_42" in publish_text,
        "collect_queries_bridge_metrics": "8016" in collect_text and "/metrics" in collect_text,
        "collect_queries_online_summary": "8015" in collect_text and "/summary" in collect_text,
        "collect_queries_controller": "8020" in collect_text and "/assignments" in collect_text,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_no_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_6_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in offensive_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = check_required_files()
    services = check_required_services()
    scripts = check_script_content()
    safety = check_no_offensive_commands()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "services": services,
        "scripts": scripts,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], services["ok"], scripts["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_6_live_mqtt_setup_validation.json"
    md_path = REPORT_DIR / "p16_6_live_mqtt_setup_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.6 Live MQTT Setup Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Required services: `{result['services']['ok']}`",
        f"- Script content: `{result['scripts']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Scope",
        "",
        "- P16.6 validates controlled MQTT JSON payloads only.",
        "- No packet capture, training, Flower, or lab scenario is started by this setup.",
        "",
    ]
    if result["files"]["missing"]:
        lines.extend(["## Missing Files", ""])
        lines.extend(f"- `{path}`" for path in result["files"]["missing"])
        lines.append("")
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
