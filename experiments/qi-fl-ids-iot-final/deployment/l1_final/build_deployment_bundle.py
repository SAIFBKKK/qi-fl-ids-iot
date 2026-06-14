from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


FINAL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = FINAL_DIR.parents[1]
BUNDLE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BUNDLE_DIR / "artifacts"

P8_RUN_DIR = FINAL_DIR / "outputs" / "qga_fedavg_flower_l1" / "alpha_0.5" / "k3" / "runs" / "run_20260508_155659"
P8_CHECKPOINT = P8_RUN_DIR / "checkpoints" / "last_global_model.pth"
P8_RUN_SUMMARY = FINAL_DIR / "outputs" / "qga_fedavg_flower_l1" / "alpha_0.5" / "k3" / "latest_run_summary.json"
P8_THRESHOLD = P8_RUN_DIR / "artifacts" / "threshold.json"
P8_MODEL_CONFIG = P8_RUN_DIR / "artifacts" / "model_config.json"
QGA_MASK_DIR = FINAL_DIR / "outputs" / "qga_feature_selection" / "final_selected_mask"
FEATURE_NAMES = FINAL_DIR / "outputs" / "artifacts" / "features" / "feature_names.json"
SCALER = FINAL_DIR / "outputs" / "artifacts" / "scalers" / "l1_binary_robust_scaler.pkl"
P13_EVAL = FINAL_DIR / "outputs" / "reports" / "p13_dashboard_model_evaluation.csv"
P12_GLOBAL = FINAL_DIR / "outputs" / "reports" / "p12_global_ablation_summary.csv"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_if_json(source: Path, target_name: str, warnings: list[str]) -> dict[str, Any]:
    if not source.exists():
        warnings.append(f"missing_json_artifact:{_relative(source)}")
        return {"packaged": False, "source": _relative(source), "path": ""}
    target = ARTIFACTS_DIR / target_name
    shutil.copy2(source, target)
    return {"packaged": True, "source": _relative(source), "path": _relative(target), "sha256": _sha256(target)}


def build_bundle() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    run_summary = _read_json(P8_RUN_SUMMARY)
    threshold = _read_json(P8_THRESHOLD)
    model_config = _read_json(P8_MODEL_CONFIG)
    qga_features = _read_json(QGA_MASK_DIR / "selected_features.json")
    qga_mask = _read_json(QGA_MASK_DIR / "feature_mask.json")
    selection_decision = _read_json(QGA_MASK_DIR / "selection_decision.json")
    feature_names = _read_json(FEATURE_NAMES) if FEATURE_NAMES.exists() else []

    model_artifact: dict[str, Any]
    if P8_CHECKPOINT.exists() and P8_CHECKPOINT.stat().st_size < 5 * 1024 * 1024:
        target = ARTIFACTS_DIR / "model.pth"
        shutil.copy2(P8_CHECKPOINT, target)
        model_artifact = {
            "status": "packaged",
            "source": _relative(P8_CHECKPOINT),
            "path": _relative(target),
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    else:
        status = "model_checkpoint_missing" if not P8_CHECKPOINT.exists() else "model_checkpoint_too_large"
        warnings.append(status)
        model_artifact = {
            "status": status,
            "source": _relative(P8_CHECKPOINT),
            "path": "",
            "size_bytes": 0,
            "sha256": "",
        }

    feature_schema_artifact = _copy_if_json(FEATURE_NAMES, "feature_names.json", warnings)
    qga_mask_artifact = _copy_if_json(QGA_MASK_DIR / "feature_mask.json", "qga_feature_mask.json", warnings)
    selected_features_artifact = _copy_if_json(QGA_MASK_DIR / "selected_features.json", "selected_features.json", warnings)
    selection_artifact = _copy_if_json(QGA_MASK_DIR / "selection_decision.json", "selection_decision.json", warnings)

    p8_eval = next((row for row in _read_csv(P13_EVAL) if row.get("model_id") == "p8_fedavg_qga_l1"), {})
    p8_report = next((row for row in _read_csv(P12_GLOBAL) if row.get("method") == "P8 FedAvg + QGA L1"), {})
    selected_indices = qga_features.get("selected_indices") or [
        index for index, value in enumerate(qga_mask.get("mask", [])) if int(value) == 1
    ]
    selected_features = qga_features.get("selected_features") or [
        feature_names[index] for index in selected_indices if isinstance(feature_names, list) and index < len(feature_names)
    ]
    primary_threshold = threshold.get("primary_threshold", 0.4)

    selected_model = {
        "model_id": "p8_fedavg_qga_l1",
        "phase": "P8",
        "method": "FedAvg + QGA",
        "task": "l1_binary",
        "selected_mask_id": qga_features.get("mask_id", "conservative_seed_42"),
        "selected_mask_source": "final_selected_mask",
        "features_count": int(qga_features.get("selected_features_count", 12)),
        "input_dim": int(model_config.get("input_dim", 12)),
        "hidden_layers": model_config.get("hidden_layers", [128, 64]),
        "output_dim": int(model_config.get("output_dim", 2)),
        "labels": {"normal": 0, "attack": 1},
        "threshold": primary_threshold,
        "checkpoint": model_artifact,
        "metrics_reference": {
            "source": _relative(P13_EVAL if p8_eval else P12_GLOBAL),
            "macro_f1": p8_eval.get("macro_f1", p8_report.get("macro_f1", "")),
            "weighted_f1": p8_eval.get("weighted_f1", p8_report.get("weighted_f1", "")),
            "attack_recall": p8_eval.get("attack_recall", p8_report.get("attack_recall", "")),
            "fpr": p8_eval.get("fpr", p8_report.get("fpr", "")),
            "accuracy": p8_eval.get("accuracy", p8_report.get("accuracy", "")),
            "metric_source": p8_eval.get("metric_source", "reported"),
        },
        "true_flower_runtime": True,
        "calibration_decision_used": True,
        "test_scaled_npz_included": False,
    }

    feature_schema = {
        "input_modes": ["selected_12_scaled", "original_28_scaled"],
        "original_feature_count": 28,
        "selected_feature_count": selected_model["features_count"],
        "selected_indices": selected_indices,
        "selected_features": selected_features,
        "all_features": feature_names if isinstance(feature_names, list) else [],
        "scaler": {
            "available_locally": SCALER.exists(),
            "packaged": False,
            "source": _relative(SCALER),
            "note": "The final API expects scaled features. The local scaler path is documented but the pickle is not packaged to keep the delivery bundle lightweight and auditable.",
        },
    }

    metrics_reference = {
        "p13_recomputed_test": p8_eval,
        "p12_reported": p8_report,
        "p8_run_summary_source": _relative(P8_RUN_SUMMARY),
        "p8_run_id": run_summary.get("run_id", "run_20260508_155659"),
    }

    registry = {
        "recommended_model_id": "p8_fedavg_qga_l1",
        "research_alternative_model_id": "p9_qifa_qga_l1",
        "models": [
            selected_model,
            {
                "model_id": "p9_qifa_qga_l1",
                "phase": "P9",
                "method": "QIFA + QGA",
                "task": "l1_binary",
                "status": "research_alternative_report_only",
                "features_count": 12,
                "selected_mask_id": "conservative_seed_42",
                "source": "experiments/qi-fl-ids-iot-final/outputs/reports/p12_global_ablation_summary.csv",
            },
        ],
    }

    manifest = {
        "phase": "P14",
        "bundle": "l1_final",
        "selected_model": "P8 FedAvg + QGA",
        "selected_model_id": "p8_fedavg_qga_l1",
        "selected_mask_id": selected_model["selected_mask_id"],
        "features_count": selected_model["features_count"],
        "input_dim": selected_model["input_dim"],
        "output_dim": selected_model["output_dim"],
        "test_scaled_npz_included": False,
        "artifacts": {
            "model": model_artifact,
            "feature_names": feature_schema_artifact,
            "qga_mask": qga_mask_artifact,
            "selected_features": selected_features_artifact,
            "selection_decision": selection_artifact,
        },
        "status": "ready" if model_artifact["status"] == "packaged" else "model_artifact_unavailable",
        "warnings": warnings,
    }

    _write_json(BUNDLE_DIR / "selected_model.json", selected_model)
    _write_json(BUNDLE_DIR / "feature_schema.json", feature_schema)
    _write_json(BUNDLE_DIR / "metrics_reference.json", metrics_reference)
    _write_json(BUNDLE_DIR / "model_registry_deployment.json", registry)
    _write_json(BUNDLE_DIR / "qga_mask_reference.json", {"mask": qga_mask, "selected_features": qga_features, "selection_decision": selection_decision})
    _write_json(BUNDLE_DIR / "deployment_manifest.json", manifest)
    return manifest


def main() -> int:
    manifest = build_bundle()
    print(json.dumps({"status": manifest["status"], "warnings": manifest["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
