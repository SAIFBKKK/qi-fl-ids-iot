from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
DASHBOARD_ROOT = FINAL_ROOT / "dashboard"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_9_2_validate_live_defense_demo_mode.py"

APP_PATH = DASHBOARD_ROOT / "app.py"
DEMO_TEMPLATE_PATH = DASHBOARD_ROOT / "templates" / "demo.html"
DEMO_JS_PATH = DASHBOARD_ROOT / "static" / "js" / "demo.js"
DOC_PATH = LIVE_LAB_ROOT / "docs" / "p16_9_2_live_defense_demo_mode.md"
RUNBOOK_PATH = LIVE_LAB_ROOT / "docs" / "p16_9_2_live_demo_terminal_runbook.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def run_python(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )


def load_dashboard_module():
    if str(DASHBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_ROOT))
    spec = importlib.util.spec_from_file_location("test_p16_9_2_dashboard_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_p16_9_2_dashboard_app"] = module
    spec.loader.exec_module(module)
    module.REQUEST_TIMEOUT_SECONDS = 0.05
    module.SERVICE_URLS = {
        "controller": ["http://127.0.0.1:8020"],
        "validator": ["http://127.0.0.1:8015"],
        "bridge": ["http://127.0.0.1:8016"],
        "api": ["http://127.0.0.1:8014"],
    }
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


def test_dashboard_routes_are_declared() -> None:
    app_text = read(APP_PATH)
    assert '@app.get("/demo"' in app_text
    assert '"/api/live-lab/demo-state"' in app_text
    assert "def build_demo_state" in app_text


def test_demo_state_endpoint_shape() -> None:
    from fastapi.testclient import TestClient

    module = load_dashboard_module()
    client = TestClient(module.app)
    page = client.get("/demo")
    response = client.get("/api/live-lab/demo-state")
    assert page.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert "platform_status" in payload
    assert "steps" in payload
    assert "devices" in payload
    assert "model_profile" in payload
    assert "recent_events" in payload


def test_demo_ui_contains_main_steps_and_panels() -> None:
    html = read(DEMO_TEMPLATE_PATH)
    js = read(DEMO_JS_PATH)
    combined = html + "\n" + js
    for term in [
        "Platform Ready",
        "Devices Connected",
        "Model Assigned",
        "Packet Window",
        "Alert Detected",
        "Zero Runtime Errors",
        "Dark mode",
    ]:
        assert term in combined
    assert "Live Device Card" in html
    assert "Model Assignment Panel" in html
    assert "Live Alert Focus Panel" in html
    assert "Live Metrics Panel" in html
    assert "What the jury is seeing" in html


def test_live_demo_docs_exist() -> None:
    assert DOC_PATH.exists()
    assert RUNBOOK_PATH.exists()
    assert "Live Defense Demo Mode" in read(DOC_PATH)
    assert "Controlled PacketWindow Publish" in read(RUNBOOK_PATH)


def test_validation_script_returns_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_9_2_live_defense_demo_mode_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_no_offensive_commands_in_p16_9_2_files() -> None:
    files = [
        APP_PATH,
        DEMO_TEMPLATE_PATH,
        DEMO_JS_PATH,
        DOC_PATH,
        RUNBOOK_PATH,
        VALIDATION_SCRIPT,
    ]
    violations = []
    for path in files:
        text = read(path).lower()
        for term in command_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append((path, term))
    assert violations == []
