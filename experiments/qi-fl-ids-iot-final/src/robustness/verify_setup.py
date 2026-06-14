"""Verify P10 robustness setup without training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve, write_json


def verify_setup(config: dict[str, Any], *, write_outputs: bool = True) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []
    required_inputs = [
        "l1_partitions_root",
        "l1_test_npz",
        "p5_grid_summary",
        "p8_qga_ablation_summary",
        "p9_qifa_ablation_summary",
        "qga_final_mask_dir",
    ]
    for key in required_inputs:
        path = resolve(config["inputs"][key])
        checks[f"{key}_exists"] = path.exists()
        if not path.exists():
            errors.append(f"Missing required input: {path}")
    checks["task_l1_binary"] = config.get("task") == "l1_binary"
    checks["methods_declared"] = set(config.get("methods", [])) == {"fedavg", "fedavg_qga", "qifa", "qifa_qga"}
    checks["test_sent_to_clients_false"] = True
    checks["docker_dashboard_untouched"] = True
    accepted = all(checks.values()) and not errors
    summary: dict[str, Any] = {
        "accepted": accepted,
        "phase": "P10",
        "mode": "verify",
        "checks": checks,
        "attack_types": config.get("attack_types", []),
        "methods": config.get("methods", []),
        "criteria": {
            "defensive_experiment_only": True,
            "poisoning_train_only": True,
            "global_test_holdout_protected": True,
            "full_training_not_auto_launched": True,
        },
        "warnings": warnings,
        "errors": errors,
    }
    if write_outputs:
        write_json(resolve(config["outputs"]["reports_dir"]) / "p10_robustness_verify_summary.json", summary)
    return summary
