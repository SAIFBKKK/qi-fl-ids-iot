from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
KALI_ROOT = LIVE_LAB_ROOT / "kali"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_10_validate_kali_readiness_setup.py"


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


def p16_10_files() -> list[Path]:
    return [
        KALI_ROOT / "README.md",
        KALI_ROOT / "kali_readiness_check.py",
        KALI_ROOT / "scenario_catalog.yaml",
        KALI_ROOT / "safe_scope.md",
        KALI_ROOT / "docs" / "kali_vm_readiness.md",
        KALI_ROOT / "docs" / "kali_network_isolation.md",
        KALI_ROOT / "docs" / "selected_scenarios.md",
        FINAL_ROOT / "outputs" / "reports" / "p16_10_kali_lab_workstation_plan.md",
        VALIDATION_SCRIPT,
    ]


def test_kali_structure_present() -> None:
    assert KALI_ROOT.exists()
    assert (KALI_ROOT / "README.md").exists()
    assert (KALI_ROOT / "kali_readiness_check.py").exists()
    assert (KALI_ROOT / "scenario_catalog.yaml").exists()
    assert (KALI_ROOT / "safe_scope.md").exists()
    assert (KALI_ROOT / "docs" / "kali_vm_readiness.md").exists()
    assert (KALI_ROOT / "docs" / "kali_network_isolation.md").exists()
    assert (KALI_ROOT / "docs" / "selected_scenarios.md").exists()


def test_scenario_catalog_contains_selected_scenarios() -> None:
    catalog = read(KALI_ROOT / "scenario_catalog.yaml")
    assert "icmp_flood_like" in catalog
    assert "tcp_syn_recon_like" in catalog
    assert "http_slow_like" in catalog
    assert "descriptive_catalog_only" in catalog
    assert "no_commands_no_scenarios_in_p16_10" in catalog


def test_safe_scope_exists_and_is_clear() -> None:
    text = read(KALI_ROOT / "safe_scope.md").lower()
    assert "isolated lab workstation" in text
    assert "does not run live packet capture" in text
    assert "does not generate traffic" in text
    assert "does not scan a network" in text


def test_validation_script_generates_ok_true() -> None:
    result = run_python(VALIDATION_SCRIPT)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_10_kali_readiness_setup_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True


def test_no_offensive_commands_ready_to_execute() -> None:
    violations = []
    for path in p16_10_files():
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for tool in reference_tools():
                if not any(pattern.search(line) for pattern in command_like_patterns(tool)):
                    continue
                allowed = (
                    "shutil.which" in line
                    or "scientific_reference_tool" in line
                    or "reference" in line.lower()
                    or "inventory" in line.lower()
                    or "cited" in line.lower()
                )
                if not allowed:
                    violations.append((path, line_no, tool, line.strip()))
    assert violations == []
