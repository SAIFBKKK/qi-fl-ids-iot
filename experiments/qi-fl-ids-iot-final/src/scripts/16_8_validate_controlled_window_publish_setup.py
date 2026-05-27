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
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

RUN_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_run_controlled_window_publish.py"
COLLECT_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_collect_window_publish_evidence.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_validate_controlled_window_publish_setup.py"
REALTIME_AGENT_ROOT = LIVE_LAB_ROOT / "realtime_agent"
DOC_PATH = LIVE_LAB_ROOT / "docs" / "p16_8_controlled_window_publish.md"
REPORT_PATH = REPORT_DIR / "p16_8_controlled_window_publish_report.md"

P16_8_FILES = [
    RUN_SCRIPT,
    COLLECT_SCRIPT,
    VALIDATION_SCRIPT,
    DOC_PATH,
    REPORT_PATH,
    REPORT_DIR / "p16_8_window_publish_evidence.json",
    REPORT_DIR / "p16_8_window_publish_evidence_table.md",
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
    required = [RUN_SCRIPT, COLLECT_SCRIPT, DOC_PATH, REPORT_PATH, REALTIME_AGENT_ROOT / "run_realtime_window_agent.py"]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def run_dry_run(node_id: str, input_mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
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
    return {
        "ok": completed.returncode == 0 and payload is not None,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "features_count": payload.get("features_count") if payload else None,
        "topic": payload.get("topic") if payload else None,
        "published": payload.get("published") if payload else None,
        "dry_run": payload.get("dry_run") if payload else None,
    }


def check_dry_runs() -> dict[str, Any]:
    vm1 = run_dry_run("iot-rpi-weak", "selected_12_scaled")
    vm2 = run_dry_run("iot-smart-watch-medium", "original_28_scaled")
    checks = {
        "vm1_selected_12_scaled": vm1["ok"] and vm1["features_count"] == 12 and vm1["dry_run"] is True,
        "vm2_original_28_scaled": vm2["ok"] and vm2["features_count"] == 28 and vm2["dry_run"] is True,
    }
    return {"ok": all(checks.values()), "checks": checks, "runs": {"vm1": vm1, "vm2": vm2}}


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_8_FILES:
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
    dry_runs = check_dry_runs()
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "dry_runs": dry_runs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], dry_runs["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_8_controlled_window_publish_validation.json"
    md_path = REPORT_DIR / "p16_8_controlled_window_publish_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.8 Controlled Window Publish Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- Dry-run checks: `{result['dry_runs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Dry-Run Feature Counts",
        "",
        "| Node | Mode | Expected | Observed | OK |",
        "|---|---|---:|---:|---|",
        (
            "| `iot-rpi-weak` | `selected_12_scaled` | 12 | "
            f"{result['dry_runs']['runs']['vm1'].get('features_count')} | "
            f"`{result['dry_runs']['checks']['vm1_selected_12_scaled']}` |"
        ),
        (
            "| `iot-smart-watch-medium` | `original_28_scaled` | 28 | "
            f"{result['dry_runs']['runs']['vm2'].get('features_count')} | "
            f"`{result['dry_runs']['checks']['vm2_original_28_scaled']}` |"
        ),
    ]
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
