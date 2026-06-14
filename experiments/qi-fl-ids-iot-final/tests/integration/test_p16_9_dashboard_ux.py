from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_9_validate_dashboard_ux.py"


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_dashboard_structure_present() -> None:
    assert (DASHBOARD_ROOT / "app.py").exists()
    assert (DASHBOARD_ROOT / "templates" / "index.html").exists()
    assert (DASHBOARD_ROOT / "static" / "css" / "dashboard.css").exists()
    assert (DASHBOARD_ROOT / "static" / "js" / "dashboard.js").exists()


def test_dashboard_components_detected() -> None:
    html = read(DASHBOARD_ROOT / "templates" / "index.html")
    assert "kpi-devices" in html
    assert "device-table-body" in html
    assert "model-id" in html
    assert "alert-event-list" in html
    assert "service-status-list" in html


def test_light_dark_theme_detected() -> None:
    html = read(DASHBOARD_ROOT / "templates" / "index.html")
    css = read(DASHBOARD_ROOT / "static" / "css" / "dashboard.css")
    theme = read(DASHBOARD_ROOT / "static" / "js" / "theme.js")
    assert "theme-toggle" in html
    assert "data-theme" in html
    assert '[data-theme="dark"]' in css
    assert "localStorage" in theme


def test_device_and_alert_logic_detected() -> None:
    js = read(DASHBOARD_ROOT / "static" / "js" / "dashboard.js")
    assert "knownNodes" in js
    assert "Device connected" in js
    assert "knownAlerts" in js
    assert "Alert detected" in js
    assert "renderDevices" in js
    assert "renderAlerts" in js


def test_backend_live_state_endpoint_detected() -> None:
    app = read(DASHBOARD_ROOT / "app.py")
    assert "/api/live-lab/state" in app
    assert "build_live_lab_state" in app
    assert "live-lab-controller" in app
    assert "final-mqtt-bridge" in app


def test_validation_script_returns_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_9_dashboard_ux_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True
