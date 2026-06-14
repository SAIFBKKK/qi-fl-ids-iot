from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

APP_PATH = DASHBOARD_ROOT / "app.py"
DEMO_TEMPLATE_PATH = DASHBOARD_ROOT / "templates" / "demo.html"
INDEX_TEMPLATE_PATH = DASHBOARD_ROOT / "templates" / "index.html"
CSS_PATH = DASHBOARD_ROOT / "static" / "css" / "dashboard.css"
DEMO_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "demo.js"
THEME_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "theme.js"
DOC_PATH = LIVE_LAB_ROOT / "docs" / "p16_9_2_live_defense_demo_mode.md"
RUNBOOK_PATH = LIVE_LAB_ROOT / "docs" / "p16_9_2_live_demo_terminal_runbook.md"
REPORT_PATH = REPORT_DIR / "p16_9_2_live_defense_demo_mode_report.md"

VALIDATION_JSON = REPORT_DIR / "p16_9_2_live_defense_demo_mode_validation.json"
VALIDATION_MD = REPORT_DIR / "p16_9_2_live_defense_demo_mode_validation.md"

P16_9_2_FILES = [
    APP_PATH,
    DEMO_TEMPLATE_PATH,
    INDEX_TEMPLATE_PATH,
    CSS_PATH,
    DEMO_JS_PATH,
    THEME_JS_PATH,
    DOC_PATH,
    RUNBOOK_PATH,
    REPORT_PATH,
    Path(__file__).resolve(),
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


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


def load_dashboard_app_module():
    os.environ.setdefault("LIVE_LAB_DASHBOARD_TIMEOUT", "0.05")
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    spec = importlib.util.spec_from_file_location("p16_9_2_dashboard_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import dashboard app")
    module = importlib.util.module_from_spec(spec)
    sys.modules["p16_9_2_dashboard_app"] = module
    spec.loader.exec_module(module)
    module.REQUEST_TIMEOUT_SECONDS = 0.05
    module.SERVICE_URLS = {
        "controller": ["http://127.0.0.1:8020"],
        "validator": ["http://127.0.0.1:8015"],
        "bridge": ["http://127.0.0.1:8016"],
        "api": ["http://127.0.0.1:8014"],
    }
    return module


def check_files() -> dict[str, Any]:
    missing = [rel(path) for path in P16_9_2_FILES if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_routes_and_endpoint() -> dict[str, Any]:
    app_text = read(APP_PATH)
    checks = {
        "demo_route_present": '@app.get("/demo"' in app_text,
        "demo_state_endpoint_present": '"/api/live-lab/demo-state"' in app_text,
        "demo_state_builder_present": "def build_demo_state" in app_text,
        "main_dashboard_link_present": 'href="/demo"' in read(INDEX_TEMPLATE_PATH),
    }
    endpoint_probe: dict[str, Any] = {"ok": False}
    try:
        from fastapi.testclient import TestClient

        module = load_dashboard_app_module()
        client = TestClient(module.app)
        demo_page = client.get("/demo")
        demo_state = client.get("/api/live-lab/demo-state")
        payload = demo_state.json() if demo_state.status_code == 200 else {}
        endpoint_probe = {
            "ok": demo_page.status_code == 200 and demo_state.status_code == 200,
            "demo_status": demo_page.status_code,
            "demo_state_status": demo_state.status_code,
            "demo_state_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "warnings": payload.get("warnings", []) if isinstance(payload, dict) else [],
        }
        checks["demo_state_shape"] = all(
            key in payload
            for key in [
                "platform_status",
                "steps",
                "devices",
                "assignments",
                "model_profile",
                "latest_alert",
                "metrics",
                "recent_events",
            ]
        )
    except Exception as exc:  # noqa: BLE001 - validation should report the import/probe failure.
        endpoint_probe = {"ok": False, "warning": str(exc)}
        checks["demo_state_shape"] = False
    return {"ok": all(checks.values()) and endpoint_probe["ok"], "checks": checks, "endpoint_probe": endpoint_probe}


def check_ui_components() -> dict[str, Any]:
    html = read(DEMO_TEMPLATE_PATH)
    css = read(CSS_PATH)
    js = read(DEMO_JS_PATH)
    theme = read(THEME_JS_PATH)
    combined = "\n".join([html, css, js, theme])
    required_terms = [
        "Platform Ready",
        "Devices Connected",
        "Model Assigned",
        "Packet Window",
        "Alert Detected",
        "Zero Runtime Errors",
        "Dark mode",
    ]
    checks = {f"term_{term.replace(' ', '_').lower()}": term in combined for term in required_terms}
    checks.update(
        {
            "live_device_cards": "demo-device-cards" in html and "Live Device Card" in html,
            "model_assignment_panel": "Model Assignment Panel" in html,
            "live_alert_focus_panel": "Live Alert Focus Panel" in html,
            "live_metrics_panel": "Live Metrics Panel" in html,
            "live_event_stream": "demo-event-stream" in html and "Live Event Stream" in html,
            "jury_explanation_panel": "What the jury is seeing" in html,
            "presentation_controls": "Refresh now" in html and "Open Grafana" in html,
            "theme_support": "localStorage" in theme and "data-theme" in html,
        }
    )
    return {"ok": all(checks.values()), "checks": checks}


def check_docs() -> dict[str, Any]:
    doc = read(DOC_PATH)
    runbook = read(RUNBOOK_PATH)
    report = read(REPORT_PATH)
    checks = {
        "live_demo_doc": DOC_PATH.exists() and "Live Defense Demo Mode" in doc,
        "runbook": RUNBOOK_PATH.exists() and "Controlled PacketWindow Publish" in runbook,
        "report": REPORT_PATH.exists() and "/api/live-lab/demo-state" in report,
        "safe_scope_doc": "does not run packet capture" in doc.lower(),
        "safe_scope_runbook": "no live capture" in runbook.lower(),
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_9_2_FILES:
        if not path.exists():
            continue
        text = read(path).lower()
        for term in command_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.9.2 Live Defense Demo Mode Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Routes and endpoint: `{result['routes_and_endpoint']['ok']}`",
        f"- UI components: `{result['ui_components']['ok']}`",
        f"- Documentation: `{result['docs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Endpoint Probe",
        "",
        f"- `/demo`: `{result['routes_and_endpoint']['endpoint_probe'].get('demo_status')}`",
        f"- `/api/live-lab/demo-state`: `{result['routes_and_endpoint']['endpoint_probe'].get('demo_state_status')}`",
        "",
        "## UI Terms",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in result["ui_components"]["checks"].items())
    if result["safety"]["violations"]:
        lines.extend(["", "## Safety Violations", ""])
        lines.extend(f"- `{item['file']}`: `{item['term']}`" for item in result["safety"]["violations"])
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def run_validation() -> dict[str, Any]:
    files = check_files()
    routes = check_routes_and_endpoint()
    ui = check_ui_components()
    docs = check_docs()
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "routes_and_endpoint": routes,
        "ui_components": ui,
        "docs": docs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], routes["ok"], ui["ok"], docs["ok"], safety["ok"]])
    write_reports(result)
    return result


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
