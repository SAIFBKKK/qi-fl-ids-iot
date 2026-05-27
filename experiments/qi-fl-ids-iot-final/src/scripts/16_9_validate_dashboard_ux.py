from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

APP_PATH = DASHBOARD_ROOT / "app.py"
TEMPLATE_PATH = DASHBOARD_ROOT / "templates" / "index.html"
CSS_PATH = DASHBOARD_ROOT / "static" / "css" / "dashboard.css"
JS_DASHBOARD_PATH = DASHBOARD_ROOT / "static" / "js" / "dashboard.js"
JS_THEME_PATH = DASHBOARD_ROOT / "static" / "js" / "theme.js"
JS_NOTIFICATIONS_PATH = DASHBOARD_ROOT / "static" / "js" / "notifications.js"
PLAN_PATH = REPORT_DIR / "p16_9_dashboard_ux_plan.md"
SCREENSHOT_CHECKLIST_PATH = REPORT_DIR / "p16_9_dashboard_screenshot_checklist.md"

P16_9_FILES = [
    APP_PATH,
    TEMPLATE_PATH,
    CSS_PATH,
    JS_DASHBOARD_PATH,
    JS_THEME_PATH,
    JS_NOTIFICATIONS_PATH,
    PLAN_PATH,
    SCREENSHOT_CHECKLIST_PATH,
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def check_files() -> dict[str, Any]:
    required = [APP_PATH, TEMPLATE_PATH, CSS_PATH, JS_DASHBOARD_PATH, JS_THEME_PATH, JS_NOTIFICATIONS_PATH, PLAN_PATH, SCREENSHOT_CHECKLIST_PATH]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_components() -> dict[str, Any]:
    app_text = read(APP_PATH)
    html = read(TEMPLATE_PATH)
    css = read(CSS_PATH)
    js = read(JS_DASHBOARD_PATH)
    theme = read(JS_THEME_PATH)
    notifications = read(JS_NOTIFICATIONS_PATH)
    checks = {
        "backend_live_state_endpoint": "/api/live-lab/state" in app_text,
        "service_aggregation": "live-lab-controller" in app_text and "online-validator" in app_text,
        "light_dark_theme": "data-theme" in html and "localStorage" in theme and "[data-theme=\"dark\"]" in css,
        "device_connected_notification": "Device connected" in js and "knownNodes" in js,
        "alert_detected_notification": "Alert detected" in js and "knownAlerts" in js,
        "device_table": "device-table-body" in html and "assigned_tier" in js,
        "model_profile_panel": "model-id" in html and "selected-mask-id" in html,
        "recent_alerts_panel": "alert-event-list" in html and "Recent Alerts" in html,
        "service_status_panel": "service-status-list" in html,
        "toast_notifications": "toast-stack" in html and "P169Notifications" in notifications,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_dashboard_backend_endpoints() -> dict[str, Any]:
    os.environ.setdefault("LIVE_LAB_DASHBOARD_TIMEOUT", "0.2")
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    try:
        from fastapi.testclient import TestClient
        import app as dashboard_app

        dashboard_app.SERVICE_URLS = {
            "controller": ["http://127.0.0.1:8020"],
            "validator": ["http://127.0.0.1:8015"],
            "bridge": ["http://127.0.0.1:8016"],
            "api": ["http://127.0.0.1:8014"],
        }

        client = TestClient(dashboard_app.app)
        health = client.get("/health")
        live_state = client.get("/api/live-lab/state")
        return {
            "ok": health.status_code == 200 and live_state.status_code == 200,
            "results": [
                {"path": "/health", "status": health.status_code},
                {"path": "/api/live-lab/state", "status": live_state.status_code},
            ],
            "note": "Validated against the local FastAPI app object; rebuild dashboard-p13 for the running container.",
        }
    except Exception as exc:  # noqa: BLE001 - validation should report why backend import failed.
        return {"ok": False, "results": [], "warning": str(exc)}


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_9_FILES:
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
    components = check_components()
    endpoints = check_dashboard_backend_endpoints()
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "components": components,
        "dashboard_backend_endpoints": endpoints,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], components["ok"], endpoints["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_9_dashboard_ux_validation.json"
    md_path = REPORT_DIR / "p16_9_dashboard_ux_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.9 Dashboard UX Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- UX components: `{result['components']['ok']}`",
        f"- Dashboard backend endpoint probe: `{result['dashboard_backend_endpoints']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Component Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in result["components"]["checks"].items())
    if result["safety"]["violations"]:
        lines.extend(["", "## Safety Violations", ""])
        lines.extend(f"- `{item['file']}`: `{item['term']}`" for item in result["safety"]["violations"])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
