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
REALTIME_ROOT = LIVE_LAB_ROOT / "realtime_agent"

PASSIVE_CAPTURE = REALTIME_ROOT / "passive_capture.py"
RUN_AGENT = REALTIME_ROOT / "run_realtime_window_agent.py"
WRAPPER = FINAL_ROOT / "src" / "scripts" / "16_12_run_passive_observation_publish.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_12_validate_passive_observation_setup.py"
DOC = LIVE_LAB_ROOT / "docs" / "p16_12_real_traffic_passive_observation.md"
REPORT = FINAL_ROOT / "outputs" / "reports" / "p16_12_real_traffic_passive_observation_report.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def run_python(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def offensive_terms() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "go" + "lang-http" + "flood",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
    ]


def command_like_patterns(term: str) -> list[re.Pattern[str]]:
    escaped = re.escape(term)
    return [
        re.compile(rf"^\s*(```)?\s*(\$|#|>)?\s*(sudo\s+)?{escaped}(\s|$)", re.IGNORECASE),
        re.compile(rf"\b(sudo|bash|sh|cmd)\s+[^`'\"]*{escaped}\b", re.IGNORECASE),
        re.compile(rf"\bpython3?\s+[^`'\"]*{escaped}\b", re.IGNORECASE),
    ]


def p16_12_files() -> list[Path]:
    return [PASSIVE_CAPTURE, RUN_AGENT, WRAPPER, VALIDATION_SCRIPT, DOC, REPORT]


def test_passive_capture_importable_and_filter_safe() -> None:
    module = load_module("test_p16_12_passive_capture", PASSIVE_CAPTURE)
    capture_filter = module.build_safe_capture_filter("192.168.56.101", "192.168.56.103")
    assert "192.168.56.101" in capture_filter
    assert "192.168.56.103" in capture_filter
    assert module.passive_packet_source("enp0s8", 30, 30, "192.168.56.101", "192.168.56.103").describe()["passive"] is True


def test_wrapper_importable() -> None:
    module = load_module("test_p16_12_wrapper", WRAPPER)
    assert module.SUPPORTED_NODES == ("iot-rpi-weak",)
    assert module.parse_args is not None


def test_source_live_requires_allow_live_capture() -> None:
    result = run_python(
        RUN_AGENT,
        "--node-id",
        "iot-rpi-weak",
        "--source",
        "live",
        "--interface",
        "enp0s8",
        "--node-ip",
        "192.168.56.101",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "--allow-live-capture" in result.stderr


def test_source_synthetic_remains_available() -> None:
    result = run_python(
        RUN_AGENT,
        "--node-id",
        "iot-rpi-weak",
        "--source",
        "synthetic",
        "--input-mode",
        "selected_12_scaled",
        "--window-size",
        "30",
        "--max-windows",
        "1",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["source"]["source"] == "synthetic"


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_12_passive_observation_setup_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_no_offensive_commands_in_p16_12_files() -> None:
    violations = []
    for path in p16_12_files():
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for term in offensive_terms():
                if any(pattern.search(line) for pattern in command_like_patterns(term)):
                    violations.append((path, line_no, term, line.strip()))
    assert violations == []
