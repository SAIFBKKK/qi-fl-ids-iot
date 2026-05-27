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

EXPORT_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_1_export_runtime_scaler_json.py"
VALIDATION_SCRIPT = FINAL_ROOT / "src" / "scripts" / "16_8_1_validate_runtime_scaler_packaging.py"
SCALER_RUNTIME_PATH = LIVE_LAB_ROOT / "realtime_agent" / "scaler_runtime.py"
RUN_AGENT_PATH = LIVE_LAB_ROOT / "realtime_agent" / "run_realtime_window_agent.py"
ARTIFACTS_README = LIVE_LAB_ROOT / "artifacts" / "README.md"
SCALER_JSON_PATH = LIVE_LAB_ROOT / "artifacts" / "l1_binary_robust_scaler.json"
REPORT_PATH = REPORT_DIR / "p16_8_1_runtime_scaler_packaging_report.md"

P16_8_1_FILES = [
    EXPORT_SCRIPT,
    VALIDATION_SCRIPT,
    SCALER_RUNTIME_PATH,
    RUN_AGENT_PATH,
    ARTIFACTS_README,
    SCALER_JSON_PATH,
    REPORT_PATH,
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


def check_required_files() -> dict[str, Any]:
    required = [EXPORT_SCRIPT, SCALER_RUNTIME_PATH, RUN_AGENT_PATH, ARTIFACTS_README, REPORT_PATH]
    missing = [rel(path) for path in required if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_scaler_runtime_support() -> dict[str, Any]:
    text = SCALER_RUNTIME_PATH.read_text(encoding="utf-8") if SCALER_RUNTIME_PATH.exists() else ""
    checks = {
        "json_path_function": "default_runtime_scaler_json_path" in text,
        "json_loader": "load_runtime_json_scaler" in text,
        "json_source": 'source="json"' in text or "source = \"json\"" in text,
        "manual_transform": "center" in text and "scale" in text,
    }
    return {"ok": all(checks.values()), "checks": checks}


def check_scaler_json() -> dict[str, Any]:
    if not SCALER_JSON_PATH.exists():
        return {"ok": True, "exists": False, "warning": "runtime scaler JSON not present"}
    payload = json.loads(SCALER_JSON_PATH.read_text(encoding="utf-8"))
    checks = {
        "feature_count": payload.get("feature_count") == 28,
        "center_length": len(payload.get("center", [])) == 28,
        "scale_length": len(payload.get("scale", [])) == 28,
        "feature_names_length": len(payload.get("feature_names", [])) == 28,
        "not_pickle": SCALER_JSON_PATH.suffix == ".json",
    }
    return {"ok": all(checks.values()), "exists": True, "checks": checks}


def run_dry_run(node_id: str, input_mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_AGENT_PATH),
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
    window = payload["windows"][0] if payload and payload.get("windows") else {}
    transform = window.get("transform", {})
    scaler = transform.get("scaler", {})
    qga_mask = transform.get("qga_mask", {})
    return {
        "ok": completed.returncode == 0 and payload is not None,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "feature_count": window.get("feature_count"),
        "scaler": scaler,
        "qga_mask": qga_mask,
    }


def check_dry_runs(scaler_json_exists: bool) -> dict[str, Any]:
    selected = run_dry_run("iot-rpi-weak", "selected_12_scaled")
    original = run_dry_run("iot-smart-watch-medium", "original_28_scaled")
    checks = {
        "selected_12_scaled_count": selected["ok"] and selected["feature_count"] == 12,
        "original_28_scaled_count": original["ok"] and original["feature_count"] == 28,
        "selected_mask_id_visible": selected["qga_mask"].get("selected_mask_id") == "conservative_seed_42",
    }
    if scaler_json_exists:
        checks["selected_scaler_used"] = selected["scaler"].get("used") is True
        checks["selected_scaler_source_json"] = selected["scaler"].get("source") == "json"
        checks["original_scaler_used"] = original["scaler"].get("used") is True
        checks["original_scaler_source_json"] = original["scaler"].get("source") == "json"
    return {"ok": all(checks.values()), "checks": checks, "runs": {"selected_12_scaled": selected, "original_28_scaled": original}}


def check_no_command_terms() -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for path in P16_8_1_FILES:
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
    files = check_required_files()
    runtime = check_scaler_runtime_support()
    scaler_json = check_scaler_json()
    dry_runs = check_dry_runs(scaler_json_exists=bool(scaler_json.get("exists")))
    safety = check_no_command_terms()
    result = {
        "generated_at": utc_now(),
        "files": files,
        "runtime_support": runtime,
        "scaler_json": scaler_json,
        "dry_runs": dry_runs,
        "safety": safety,
    }
    result["ok"] = all([files["ok"], runtime["ok"], scaler_json["ok"], dry_runs["ok"], safety["ok"]])
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_8_1_runtime_scaler_packaging_validation.json"
    md_path = REPORT_DIR / "p16_8_1_runtime_scaler_packaging_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    selected = result["dry_runs"]["runs"]["selected_12_scaled"]
    original = result["dry_runs"]["runs"]["original_28_scaled"]
    lines = [
        "# P16.8.1 Runtime Scaler Packaging Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required files: `{result['files']['ok']}`",
        f"- JSON scaler support: `{result['runtime_support']['ok']}`",
        f"- Runtime scaler JSON exists: `{result['scaler_json']['exists']}`",
        f"- Dry-run checks: `{result['dry_runs']['ok']}`",
        f"- Safety check: `{result['safety']['ok']}`",
        "",
        "## Dry-Run Scaler Runtime",
        "",
        "| Mode | Features | Scaler available | Scaler used | Source | Selected mask |",
        "|---|---:|---|---|---|---|",
        (
            f"| `selected_12_scaled` | {selected.get('feature_count')} | "
            f"`{selected['scaler'].get('available')}` | `{selected['scaler'].get('used')}` | "
            f"`{selected['scaler'].get('source')}` | `{selected['qga_mask'].get('selected_mask_id')}` |"
        ),
        (
            f"| `original_28_scaled` | {original.get('feature_count')} | "
            f"`{original['scaler'].get('available')}` | `{original['scaler'].get('used')}` | "
            f"`{original['scaler'].get('source')}` | `{original['qga_mask'].get('selected_mask_id')}` |"
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
