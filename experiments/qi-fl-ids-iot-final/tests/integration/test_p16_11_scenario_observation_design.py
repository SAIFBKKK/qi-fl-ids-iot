from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
OBS_ROOT = REPO_ROOT / "experiments" / "live_lab" / "scenario_observation"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_11_validate_scenario_observation_design.py"

EXPECTED_SCENARIOS = ["icmp_flood_like", "tcp_syn_recon_like", "http_slow_like"]


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


def reference_tools() -> list[str]:
    return [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "go" + "lang-http" + "flood",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
    ]


def command_like_patterns(tool: str) -> list[re.Pattern[str]]:
    escaped = re.escape(tool)
    return [
        re.compile(rf"^\s*(```)?\s*(\$|#|>)?\s*(sudo\s+)?{escaped}(\s|$)", re.IGNORECASE),
        re.compile(rf"\b(sudo|bash|sh|cmd)\s+[^`'\"]*{escaped}\b", re.IGNORECASE),
        re.compile(rf"\bpython3?\s+[^`'\"]*{escaped}\b", re.IGNORECASE),
    ]


def p16_11_files() -> list[Path]:
    return [
        OBS_ROOT / "README.md",
        OBS_ROOT / "scenario_observation_catalog.yaml",
        OBS_ROOT / "feature_expectations.yaml",
        OBS_ROOT / "observation_evidence_template.md",
        OBS_ROOT / "safe_observation_scope.md",
        OBS_ROOT / "docs" / "p16_11_controlled_scenario_observation_design.md",
        OBS_ROOT / "docs" / "scenario_to_feature_mapping.md",
        OBS_ROOT / "docs" / "dashboard_observation_flow.md",
        OBS_ROOT / "docs" / "future_activation_boundaries.md",
        FINAL_ROOT / "outputs" / "reports" / "p16_11_controlled_scenario_observation_design_report.md",
        VALIDATION_SCRIPT,
    ]


def test_structure_present() -> None:
    assert OBS_ROOT.exists()
    for path in p16_11_files():
        assert path.exists(), path


def test_catalog_contains_three_design_only_scenarios() -> None:
    catalog = read(OBS_ROOT / "scenario_observation_catalog.yaml")
    for scenario in EXPECTED_SCENARIOS:
        assert scenario in catalog
    assert catalog.count('activation_status: "design_only"') == 3
    assert "no_commands_no_activation_no_live_capture" in catalog


def test_feature_expectations_contains_expected_features() -> None:
    text = read(OBS_ROOT / "feature_expectations.yaml")
    for term in ["ICMP", "Rate", "IAT", "Number"]:
        assert term in text
    for term in ["TCP", "syn_flag_number", "syn_count"]:
        assert term in text
    for term in ["HTTP", "Duration"]:
        assert term in text
    assert "not a scientific reproduction" in text
    assert "P12/P13" in text


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_11_scenario_observation_design_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_no_offensive_commands_ready_to_execute() -> None:
    violations = []
    for path in p16_11_files():
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for tool in reference_tools():
                if not any(pattern.search(line) for pattern in command_like_patterns(tool)):
                    continue
                allowed = (
                    "reference" in line.lower()
                    or "scientific_reference_tools" in line
                    or "tool names" in line.lower()
                    or "references only" in line.lower()
                )
                if not allowed:
                    violations.append((path, line_no, tool, line.strip()))
    assert violations == []
