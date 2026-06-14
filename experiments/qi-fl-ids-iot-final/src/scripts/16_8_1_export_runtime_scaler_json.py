from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

SOURCE_PICKLE_PATH = FINAL_ROOT / "outputs" / "artifacts" / "scalers" / "l1_binary_robust_scaler.pkl"
FEATURE_NAMES_PATH = FINAL_ROOT / "outputs" / "artifacts" / "features" / "feature_names.json"
OUTPUT_JSON_PATH = LIVE_LAB_ROOT / "artifacts" / "l1_binary_robust_scaler.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def load_pickle_scaler(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def array_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def load_feature_names() -> list[str]:
    value = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 28:
        raise ValueError(f"expected 28 feature names in {FEATURE_NAMES_PATH}")
    return [str(item) for item in value]


def build_runtime_scaler_payload(scaler: Any, feature_names: list[str]) -> dict[str, Any]:
    center = array_to_float_list(getattr(scaler, "center_", []))
    scale = array_to_float_list(getattr(scaler, "scale_", []))
    if len(center) != 28 or len(scale) != 28:
        raise ValueError("source scaler must expose center_ and scale_ arrays with length 28")
    return {
        "schema_version": "1.0",
        "scaler_type": type(scaler).__name__,
        "task": "l1_binary",
        "feature_count": 28,
        "feature_names": feature_names,
        "center": center,
        "scale": scale,
        "source_pickle_path": rel(SOURCE_PICKLE_PATH),
        "created_at": utc_now(),
        "note": "Runtime JSON export for live lab VMs; pickle is not committed.",
    }


def write_warning_report(message: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": utc_now(),
        "ok": False,
        "warning": message,
        "source_pickle_path": rel(SOURCE_PICKLE_PATH),
        "output_json_path": rel(OUTPUT_JSON_PATH),
    }
    (REPORT_DIR / "p16_8_1_runtime_scaler_export_warning.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "p16_8_1_runtime_scaler_export_warning.md").write_text(
        "\n".join(
            [
                "# P16.8.1 Runtime Scaler Export Warning",
                "",
                f"- Generated at: `{report['generated_at']}`",
                f"- Warning: {message}",
                f"- Source pickle: `{report['source_pickle_path']}`",
                f"- Output JSON: `{report['output_json_path']}`",
            ]
        ),
        encoding="utf-8",
    )


def export_runtime_scaler_json() -> dict[str, Any]:
    if not SOURCE_PICKLE_PATH.exists():
        message = f"source scaler pickle not found: {SOURCE_PICKLE_PATH}"
        write_warning_report(message)
        return {
            "ok": False,
            "warning": message,
            "created": False,
            "output_json_path": rel(OUTPUT_JSON_PATH),
        }
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(f"feature names file not found: {FEATURE_NAMES_PATH}")

    scaler = load_pickle_scaler(SOURCE_PICKLE_PATH)
    payload = build_runtime_scaler_payload(scaler, load_feature_names())
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "created": True,
        "output_json_path": rel(OUTPUT_JSON_PATH),
        "feature_count": payload["feature_count"],
        "scaler_type": payload["scaler_type"],
    }


def main() -> int:
    result = export_runtime_scaler_json()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] or not result.get("created") else 1


if __name__ == "__main__":
    raise SystemExit(main())
