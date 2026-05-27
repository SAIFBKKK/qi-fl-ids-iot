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
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
VALIDATOR_ROOT = FINAL_ROOT / "deployment" / "online_validator"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

APP_PATH = DASHBOARD_ROOT / "app.py"
DASHBOARD_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "dashboard.js"
VALIDATOR_METRICS_PATH = VALIDATOR_ROOT / "metrics.py"
REPORT_PATH = REPORT_DIR / "p16_9_1_dashboard_alert_parsing_fix.md"

P16_9_1_FILES = [APP_PATH, DASHBOARD_JS_PATH, VALIDATOR_METRICS_PATH, REPORT_PATH, Path(__file__).resolve()]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def load_dashboard_app_module():
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    spec = importlib.util.spec_from_file_location("p16_9_1_dashboard_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import dashboard app")
    module = importlib.util.module_from_spec(spec)
    sys.modules["p16_9_1_dashboard_app"] = module
    spec.loader.exec_module(module)
    return module


def command_terms() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "go" + "lang-http" + "flood",
    ]


def check_files() -> dict[str, Any]:
    missing = [rel(path) for path in P16_9_1_FILES if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_source_changes() -> dict[str, Any]:
    app_text = read(APP_PATH)
    js_text = read(DASHBOARD_JS_PATH)
    validator_text = read(VALIDATOR_METRICS_PATH)
    checks = {
        "dashboard_prefers_structured_payload": "sample_payload" in app_text and 'sample.get("payload")' in app_text,
        "dashboard_label_fallbacks": "predicted_label" in app_text and "prediction_label" in app_text,
        "dashboard_confidence_fallbacks": "probability_attack" in app_text and "attack_probability" in app_text,
        "frontend_label_fallbacks": "function alertLabel" in js_text and "prediction_label" in js_text,
        "frontend_confidence_fallbacks": "function alertConfidence" in js_text and "probability_attack" in js_text,
        "validator_structured_payload_samples": 'sample["payload"] = payload' in validator_text,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_alert_extraction() -> dict[str, Any]:
    module = load_dashboard_app_module()
    summary = {
        "samples": [
            {
                "topic": "ids/alerts/iot-smart-watch-medium",
                "family": "alerts",
                "flow_id": "p16-7-window-demo",
                "timestamp": "2026-05-27T15:00:00Z",
                "received_at_unix": 1.0,
                "payload_preview": '{"schema_version":"1.0"',
                "payload": {
                    "event_type": "final_ids_alert",
                    "node_id": "iot-smart-watch-medium",
                    "timestamp": "2026-05-27T15:00:00Z",
                    "flow_id": "p16-7-window-demo",
                    "predicted_label": "attack",
                    "predicted_label_id": 1,
                    "confidence": 0.99,
                    "severity": "critical",
                },
            }
        ]
    }
    alerts = module.extract_recent_alerts(summary)
    alert = alerts[0] if alerts else {}
    checks = {
        "alert_found": bool(alerts),
        "label_attack": alert.get("predicted_label") == "attack",
        "confidence_099": alert.get("confidence") == 0.99,
        "severity_critical": alert.get("severity") == "critical",
        "flow_id_preserved": alert.get("flow_id") == "p16-7-window-demo",
        "structured_status": alert.get("payload_parse_status") == "structured",
    }
    return {"ok": all(checks.values()), "checks": checks, "alert": alert}


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_9_1_FILES:
        if not path.exists():
            continue
        text = read(path).lower()
        for term in command_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = check_files()
    source = check_source_changes()
    extraction = check_alert_extraction()
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "source_changes": source,
        "alert_extraction": extraction,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], source["ok"], extraction["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_9_1_dashboard_alert_parsing_validation.json"
    md_path = REPORT_DIR / "p16_9_1_dashboard_alert_parsing_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    alert = result["alert_extraction"].get("alert", {})
    lines = [
        "# P16.9.1 Dashboard Alert Parsing Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Source changes: `{result['source_changes']['ok']}`",
        f"- Alert extraction: `{result['alert_extraction']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Synthetic Alert Extraction",
        "",
        f"- Node: `{alert.get('node_id')}`",
        f"- Severity: `{alert.get('severity')}`",
        f"- Label: `{alert.get('predicted_label')}`",
        f"- Confidence: `{alert.get('confidence')}`",
        f"- Flow: `{alert.get('flow_id')}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
