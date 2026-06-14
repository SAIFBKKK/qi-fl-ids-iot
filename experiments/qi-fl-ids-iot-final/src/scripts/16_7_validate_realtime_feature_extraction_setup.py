from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REALTIME_AGENT_ROOT = LIVE_LAB_ROOT / "realtime_agent"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

CLI_PATH = REALTIME_AGENT_ROOT / "run_realtime_window_agent.py"
DOC_PATH = LIVE_LAB_ROOT / "docs" / "p16_7_realtime_feature_extraction.md"
PLAN_PATH = REPORT_DIR / "p16_7_realtime_feature_extraction_plan.md"
FEATURE_NAMES_PATH = FINAL_ROOT / "outputs" / "artifacts" / "features" / "feature_names.json"
QGA_MASK_PATH = FINAL_ROOT / "outputs" / "qga_feature_selection" / "final_selected_mask" / "feature_mask.json"

REQUIRED_REALTIME_FILES = [
    REALTIME_AGENT_ROOT / "packet_capture.py",
    REALTIME_AGENT_ROOT / "flow_window.py",
    REALTIME_AGENT_ROOT / "feature_extractor.py",
    REALTIME_AGENT_ROOT / "scaler_runtime.py",
    REALTIME_AGENT_ROOT / "edge_inference.py",
    REALTIME_AGENT_ROOT / "mqtt_runtime.py",
    CLI_PATH,
    REALTIME_AGENT_ROOT / "README.md",
]

P16_7_FILES = [
    *REQUIRED_REALTIME_FILES,
    DOC_PATH,
    PLAN_PATH,
    Path(__file__).resolve(),
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


def check_files() -> dict[str, Any]:
    required = [*REQUIRED_REALTIME_FILES, DOC_PATH, PLAN_PATH]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_artifacts() -> dict[str, Any]:
    return {
        "ok": FEATURE_NAMES_PATH.exists() and QGA_MASK_PATH.exists(),
        "feature_names_path": rel(FEATURE_NAMES_PATH),
        "feature_names_found": FEATURE_NAMES_PATH.exists(),
        "qga_mask_path": rel(QGA_MASK_PATH),
        "qga_mask_found": QGA_MASK_PATH.exists(),
    }


def run_cli(input_mode: str, node_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--node-id",
            node_id,
            "--input-mode",
            input_mode,
            "--window-size",
            "30",
            "--max-windows",
            "1",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    payload: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    feature_count = None
    if payload and payload.get("windows"):
        feature_count = payload["windows"][0].get("feature_count")
    return {
        "ok": completed.returncode == 0 and payload is not None,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "feature_count": feature_count,
        "payload_ok": payload.get("ok") if payload else None,
    }


def check_dry_runs() -> dict[str, Any]:
    selected = run_cli("selected_12_scaled", "iot-drone-sitl")
    scaled = run_cli("original_28_scaled", "iot-smart-watch-medium")
    unscaled = run_cli("original_28_unscaled", "iot-smart-watch-medium")
    checks = {
        "selected_12_scaled": selected["ok"] and selected["feature_count"] == 12,
        "original_28_scaled": scaled["ok"] and scaled["feature_count"] == 28,
        "original_28_unscaled": unscaled["ok"] and unscaled["feature_count"] == 28,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "runs": {
            "selected_12_scaled": selected,
            "original_28_scaled": scaled,
            "original_28_unscaled": unscaled,
        },
    }


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_7_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in command_terms():
            pattern = rf"(^|\s)(sudo\s+|python3?\s+|bash\s+|sh\s+|cmd\s+/c\s+)?{re.escape(term)}(\s|$)"
            if re.search(pattern, text):
                violations.append({"file": rel(path), "term": term})
    return {"ok": not violations, "violations": violations}


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = check_files()
    artifacts = check_artifacts()
    dry_runs = check_dry_runs()
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "artifacts": artifacts,
        "dry_runs": dry_runs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], artifacts["ok"], dry_runs["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_7_realtime_feature_extraction_validation.json"
    md_path = REPORT_DIR / "p16_7_realtime_feature_extraction_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.7 Realtime Feature Extraction Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Realtime files: `{result['files']['ok']}`",
        f"- Feature schema found: `{result['artifacts']['feature_names_found']}`",
        f"- QGA mask found: `{result['artifacts']['qga_mask_found']}`",
        f"- Dry-run CLI checks: `{result['dry_runs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Dry-Run Feature Counts",
        "",
        "| Mode | Expected | Observed | OK |",
        "|---|---:|---:|---|",
    ]
    expected = {
        "selected_12_scaled": 12,
        "original_28_scaled": 28,
        "original_28_unscaled": 28,
    }
    for mode, run in result["dry_runs"]["runs"].items():
        lines.append(
            f"| `{mode}` | {expected[mode]} | {run.get('feature_count')} | `{result['dry_runs']['checks'][mode]}` |"
        )
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

