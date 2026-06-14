from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
OBS_ROOT = LIVE_LAB_ROOT / "scenario_observation"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

README_PATH = OBS_ROOT / "README.md"
CATALOG_PATH = OBS_ROOT / "scenario_observation_catalog.yaml"
FEATURE_EXPECTATIONS_PATH = OBS_ROOT / "feature_expectations.yaml"
EVIDENCE_TEMPLATE_PATH = OBS_ROOT / "observation_evidence_template.md"
SAFE_SCOPE_PATH = OBS_ROOT / "safe_observation_scope.md"
DOC_DESIGN_PATH = OBS_ROOT / "docs" / "p16_11_controlled_scenario_observation_design.md"
DOC_MAPPING_PATH = OBS_ROOT / "docs" / "scenario_to_feature_mapping.md"
DOC_DASHBOARD_PATH = OBS_ROOT / "docs" / "dashboard_observation_flow.md"
DOC_BOUNDARIES_PATH = OBS_ROOT / "docs" / "future_activation_boundaries.md"
REPORT_PATH = REPORT_DIR / "p16_11_controlled_scenario_observation_design_report.md"
VALIDATION_JSON = REPORT_DIR / "p16_11_scenario_observation_design_validation.json"
VALIDATION_MD = REPORT_DIR / "p16_11_scenario_observation_design_validation.md"

P16_11_FILES = [
    README_PATH,
    CATALOG_PATH,
    FEATURE_EXPECTATIONS_PATH,
    EVIDENCE_TEMPLATE_PATH,
    SAFE_SCOPE_PATH,
    DOC_DESIGN_PATH,
    DOC_MAPPING_PATH,
    DOC_DASHBOARD_PATH,
    DOC_BOUNDARIES_PATH,
    REPORT_PATH,
    Path(__file__).resolve(),
]

EXPECTED_SCENARIOS = ["icmp_flood_like", "tcp_syn_recon_like", "http_slow_like"]
EXPECTED_FEATURES = {
    "icmp_flood_like": ["ICMP", "Rate", "IAT", "Number"],
    "tcp_syn_recon_like": ["TCP", "syn_flag_number", "syn_count", "Rate"],
    "http_slow_like": ["HTTP", "TCP", "Duration", "IAT", "Rate"],
}


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
    missing = [rel(path) for path in P16_11_FILES if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_catalog() -> dict[str, Any]:
    text = read(CATALOG_PATH)
    checks: dict[str, bool] = {
        "catalog_present": CATALOG_PATH.exists(),
        "scope_design_only": "controlled_scenario_observation_design_only" in text,
        "execution_policy": "no_commands_no_activation_no_live_capture" in text,
        "activation_status_count": text.count('activation_status: "design_only"') == 3,
    }
    for scenario in EXPECTED_SCENARIOS:
        checks[f"scenario_{scenario}"] = scenario in text
    for field in [
        "scenario_id",
        "ciciot_family",
        "scientific_reference_tools",
        "intended_lab_role",
        "expected_packet_pattern",
        "expected_feature_shift",
        "target_observation_node",
        "mqtt_topic_expected",
        "dashboard_expected_evidence",
        "safety_boundary",
    ]:
        checks[f"field_{field}"] = field in text
    return {"ok": all(checks.values()), "checks": checks}


def check_feature_expectations() -> dict[str, Any]:
    text = read(FEATURE_EXPECTATIONS_PATH)
    checks: dict[str, bool] = {
        "feature_expectations_present": FEATURE_EXPECTATIONS_PATH.exists(),
        "tendency_note": "expected live-lab feature tendencies" in text,
        "not_scientific_reproduction": "not a scientific reproduction" in text,
        "p12_p13_note": "P12/P13" in text,
    }
    for scenario, features in EXPECTED_FEATURES.items():
        checks[f"features_{scenario}"] = scenario in text and all(feature in text for feature in features)
    return {"ok": all(checks.values()), "checks": checks}


def check_docs() -> dict[str, Any]:
    design = read(DOC_DESIGN_PATH)
    mapping = read(DOC_MAPPING_PATH)
    dashboard = read(DOC_DASHBOARD_PATH)
    boundaries = read(DOC_BOUNDARIES_PATH)
    template = read(EVIDENCE_TEMPLATE_PATH)
    safe_scope = read(SAFE_SCOPE_PATH)
    report = read(REPORT_PATH)
    checks = {
        "readme": README_PATH.exists() and "P16.11" in read(README_PATH),
        "evidence_template": EVIDENCE_TEMPLATE_PATH.exists() and "PacketWindow summary" in template,
        "safe_scope": SAFE_SCOPE_PATH.exists() and "design-only phase" in safe_scope and "Kali is ready but not activated" in safe_scope,
        "design_doc": DOC_DESIGN_PATH.exists() and "Kali scenario label -> observation window -> features -> MQTT -> IDS -> dashboard alert" in design,
        "mapping_doc": DOC_MAPPING_PATH.exists() and all(scenario in mapping for scenario in EXPECTED_SCENARIOS),
        "dashboard_flow_doc": DOC_DASHBOARD_PATH.exists() and "/demo" in dashboard and "zero" in dashboard.lower(),
        "future_boundaries_doc": DOC_BOUNDARIES_PATH.exists() and "P16.11 does not activate scenarios" in boundaries,
        "report": REPORT_PATH.exists() and "P16.12 controlled observation dry-run labels" in report,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_no_executable_offensive_commands() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_11_FILES:
        if not path.exists():
            continue
        for line_no, line in enumerate(read(path).splitlines(), start=1):
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
                    violations.append({"file": rel(path), "line": str(line_no), "term": tool, "text": line.strip()})
    return {"ok": not violations, "violations": violations}


def write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.11 Scenario Observation Design Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Catalog: `{result['catalog']['ok']}`",
        f"- Feature expectations: `{result['feature_expectations']['ok']}`",
        f"- Documentation: `{result['docs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Scenarios",
        "",
    ]
    lines.extend(f"- `{scenario}`: `{result['catalog']['checks'].get(f'scenario_{scenario}')}`" for scenario in EXPECTED_SCENARIOS)
    if result["safety"]["violations"]:
        lines.extend(["", "## Safety Violations", ""])
        lines.extend(
            f"- `{item['file']}` line `{item['line']}` term `{item['term']}`"
            for item in result["safety"]["violations"]
        )
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def run_validation() -> dict[str, Any]:
    files = check_files()
    catalog = check_catalog()
    feature_expectations = check_feature_expectations()
    docs = check_docs()
    safety = check_no_executable_offensive_commands()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "catalog": catalog,
        "feature_expectations": feature_expectations,
        "docs": docs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], catalog["ok"], feature_expectations["ok"], docs["ok"], safety["ok"]])
    write_reports(result)
    return result


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
