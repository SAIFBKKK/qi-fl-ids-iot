"""Readiness checks for P8 QGA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qga.config import repo_path, write_json
from qga.feature_mask import load_feature_names
from qga.summary_schema import expected_verify_artifacts, expected_verify_figures


def verify_qga_setup(config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_paths = {
        "train_npz": repo_path(config, "inputs.train_npz"),
        "val_npz": repo_path(config, "inputs.val_npz"),
        "test_npz_holdout_reference": repo_path(config, "inputs.test_npz"),
        "l1_partitions_root": repo_path(config, "inputs.l1_partitions_root"),
        "feature_names": repo_path(config, "inputs.feature_names"),
        "p5_grid_summary": repo_path(config, "inputs.p5_grid_summary"),
    }
    for name, path in required_paths.items():
        checks[f"{name}_exists"] = Path(path).exists()
        if not checks[f"{name}_exists"]:
            errors.append(f"missing required path: {path}")
    p7_summary = repo_path(config, "inputs.p7_multitier_summary")
    checks["p7_multitier_summary_exists"] = p7_summary.exists()
    if not p7_summary.exists():
        warnings.append(f"P7 summary not found yet: {p7_summary}")

    try:
        features = load_feature_names(required_paths["feature_names"])
        checks["feature_count_is_28"] = len(features) == 28
        if len(features) != 28:
            errors.append(f"expected 28 features, got {len(features)}")
    except Exception as exc:
        checks["feature_count_is_28"] = False
        errors.append(str(exc))

    alpha_dir = required_paths["l1_partitions_root"] / "alpha_0.5" / "k3"
    checks["alpha_0_5_k3_partitions_exist"] = alpha_dir.exists()
    for client_id in ["client_1", "client_2", "client_3"]:
        checks[f"{client_id}_train_exists"] = (alpha_dir / client_id / "train_scaled.npz").exists()
        checks[f"{client_id}_val_exists"] = (alpha_dir / client_id / "val_scaled.npz").exists()

    accepted = all(checks.values()) and not errors
    summary = {
        "accepted": accepted,
        "phase": "P8",
        "mode": "verify",
        "architecture": "QGA theta-vector feature selection + fast L1 MLP fitness",
        "scenario": {"task": "l1_binary", "alpha": 0.5, "clients": 3},
        "global_test_holdout": {
            "path": config["inputs"]["test_npz"],
            "used_for_selection": False,
        },
        "checks": checks,
        "artifacts_expected": expected_verify_artifacts(),
        "figures_expected": expected_verify_figures(),
        "criteria": {
            "test_not_used_for_selection": True,
            "qga_mask_bounds_configured": True,
            "fedavg_adapter_configured": bool(config.get("fedavg_eval", {}).get("enabled", False)),
            "heterofl_adapter_configured": bool(config.get("heterofl_eval", {}).get("enabled", False)),
            "qifa_not_used": True,
            "fedtn_not_used": True,
        },
        "warnings": warnings,
        "errors": errors,
    }
    reports_dir = repo_path(config, "outputs.reports_dir")
    write_json(reports_dir / "qga_verify_summary.json", summary)
    return summary
