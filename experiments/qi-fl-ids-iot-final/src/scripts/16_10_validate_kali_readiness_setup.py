from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
KALI_ROOT = LIVE_LAB_ROOT / "kali"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

README_PATH = KALI_ROOT / "README.md"
READINESS_SCRIPT_PATH = KALI_ROOT / "kali_readiness_check.py"
SCENARIO_CATALOG_PATH = KALI_ROOT / "scenario_catalog.yaml"
SAFE_SCOPE_PATH = KALI_ROOT / "safe_scope.md"
DOC_READINESS_PATH = KALI_ROOT / "docs" / "kali_vm_readiness.md"
DOC_ISOLATION_PATH = KALI_ROOT / "docs" / "kali_network_isolation.md"
DOC_SCENARIOS_PATH = KALI_ROOT / "docs" / "selected_scenarios.md"
PLAN_PATH = REPORT_DIR / "p16_10_kali_lab_workstation_plan.md"
VALIDATION_JSON = REPORT_DIR / "p16_10_kali_readiness_setup_validation.json"
VALIDATION_MD = REPORT_DIR / "p16_10_kali_readiness_setup_validation.md"

P16_10_FILES = [
    README_PATH,
    READINESS_SCRIPT_PATH,
    SCENARIO_CATALOG_PATH,
    SAFE_SCOPE_PATH,
    DOC_READINESS_PATH,
    DOC_ISOLATION_PATH,
    DOC_SCENARIOS_PATH,
    PLAN_PATH,
    Path(__file__).resolve(),
]

EXPECTED_SCENARIOS = ["icmp_flood_like", "tcp_syn_recon_like", "http_slow_like"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


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


def check_files() -> dict[str, Any]:
    missing = [rel(path) for path in P16_10_FILES if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_scenarios() -> dict[str, Any]:
    catalog = read(SCENARIO_CATALOG_PATH)
    docs = read(DOC_SCENARIOS_PATH)
    plan = read(PLAN_PATH)
    checks = {
        scenario: scenario in catalog and scenario in docs and scenario in plan
        for scenario in EXPECTED_SCENARIOS
    }
    checks.update(
        {
            "catalog_descriptive_only": "descriptive_catalog_only" in catalog,
            "no_command_policy": "no_commands_no_scenarios_in_p16_10" in catalog,
            "expected_feature_terms": all(
                term in catalog
                for term in ["ICMP", "Rate", "IAT", "Number", "TCP", "syn_flag_number", "syn_count", "HTTP", "Duration"]
            ),
        }
    )
    return {"ok": all(checks.values()), "checks": checks}


def check_readiness_script() -> dict[str, Any]:
    text = read(READINESS_SCRIPT_PATH)
    checks = {
        "reads_hostname": "socket.gethostname" in text,
        "expected_kali_ip": "192.168.56.103" in text,
        "fixed_lab_hosts": "192.168.56.1" in text and "192.168.56.101" in text and "192.168.56.102" in text,
        "server_endpoints": "8020/health" in text and "8014/ready" in text and "8016/ready" in text and "8015/ready" in text and "8013/health" in text,
        "tool_inventory_only": "inventory_only" in text and "executed" in text and "shutil.which" in text,
        "writes_runtime_reports": "p16_10_kali_readiness.json" in text and "p16_10_kali_readiness.md" in text,
        "windows_warning": "running on Windows" in text,
        "no_subnet_scan_flag": "no_subnet_scan" in text,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_docs() -> dict[str, Any]:
    safe_scope = read(SAFE_SCOPE_PATH).lower()
    isolation = read(DOC_ISOLATION_PATH).lower()
    readiness = read(DOC_READINESS_PATH).lower()
    checks = {
        "safe_scope": SAFE_SCOPE_PATH.exists() and "no scenario" in safe_scope and "no live capture" in safe_scope,
        "vm_readiness": DOC_READINESS_PATH.exists() and "192.168.56.103" in readiness,
        "network_isolation": DOC_ISOLATION_PATH.exists() and "host-only" in isolation and "do not scan" in isolation,
        "selected_scenarios_doc": DOC_SCENARIOS_PATH.exists() and all(scenario in read(DOC_SCENARIOS_PATH) for scenario in EXPECTED_SCENARIOS),
        "plan": PLAN_PATH.exists() and "P16.11 controlled scenario observation design" in read(PLAN_PATH),
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_no_executable_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_10_FILES:
        if not path.exists():
            continue
        lines = read(path).splitlines()
        for line_no, line in enumerate(lines, start=1):
            for tool in reference_tools():
                if not any(pattern.search(line) for pattern in command_like_patterns(tool)):
                    continue
                allowed_inventory_line = "shutil.which" in line or '"present"' in line or "scientific_reference_tool" in line
                allowed_doc_line = "reference" in line.lower() or "inventory" in line.lower() or "cited" in line.lower()
                if allowed_inventory_line or allowed_doc_line:
                    continue
                violations.append({"file": rel(path), "line": str(line_no), "term": tool, "text": line.strip()})
    return {"ok": not violations, "violations": violations}


def write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.10 Kali Readiness Setup Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Scenarios documented: `{result['scenarios']['ok']}`",
        f"- Readiness script: `{result['readiness_script']['ok']}`",
        f"- Documentation: `{result['docs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Scenario Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in result["scenarios"]["checks"].items())
    if result["safety"]["violations"]:
        lines.extend(["", "## Safety Violations", ""])
        lines.extend(
            f"- `{item['file']}` line `{item['line']}` term `{item['term']}`"
            for item in result["safety"]["violations"]
        )
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def run_validation() -> dict[str, Any]:
    files = check_files()
    scenarios = check_scenarios()
    readiness_script = check_readiness_script()
    docs = check_docs()
    safety = check_no_executable_offensive_commands()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "scenarios": scenarios,
        "readiness_script": readiness_script,
        "docs": docs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], scenarios["ok"], readiness_script["ok"], docs["ok"], safety["ok"]])
    write_reports(result)
    return result


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
