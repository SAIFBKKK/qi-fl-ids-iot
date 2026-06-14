"""Verify P11 FedTN/MPS setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import read_json, resolve, write_json
from .compression import estimate_low_rank_compression
from .evaluation import checkpoint_available


def verify_setup(config: dict[str, Any], *, write_outputs: bool = True) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []
    for key in [
        "qga_config",
        "qifa_config",
        "p8_qga_ablation_summary",
        "p9_qifa_ablation_summary",
        "p10_robustness_full_summary",
        "qga_final_mask_dir",
    ]:
        path = resolve(config["inputs"][key])
        checks[f"{key}_exists"] = path.exists()
        if not path.exists():
            errors.append(f"Missing input: {path}")
    mask_dir = resolve(config["inputs"]["qga_final_mask_dir"])
    decision_path = mask_dir / "selection_decision.json"
    if decision_path.exists():
        decision = read_json(decision_path)
        checks["selected_mask_id_matches"] = decision.get("selected_mask_id") == config.get("selected_mask_id")
        if not checks["selected_mask_id_matches"]:
            errors.append("Selected mask id does not match conservative_seed_42.")
    else:
        checks["selected_mask_id_matches"] = False
        errors.append(f"Missing selection decision: {decision_path}")
    checks["base_models_declared"] = set(config.get("base_models", [])) == {"fedavg_qga", "qifa_qga"}
    checks["input_dim_12"] = int(config.get("input_dim", 0)) == 12
    for key in ["fedavg_qga_checkpoint", "qifa_qga_checkpoint"]:
        path = resolve(config["inputs"].get(key, ""))
        if not checkpoint_available(path):
            warnings.append(f"{key}_checkpoint_not_available_metric_evaluation_will_be_skipped")
    estimates = [estimate_low_rank_compression(config, rank=int(rank)).to_dict() for rank in config["compression"]["ranks"]]
    accepted = all(checks.values()) and not errors
    summary: dict[str, Any] = {
        "accepted": accepted,
        "phase": "P11",
        "mode": "verify",
        "checks": checks,
        "estimates": estimates,
        "criteria": {
            "l1_only": True,
            "post_training_compression": True,
            "full_fl_not_auto_launched": True,
            "dashboard_not_modified": True,
            "docker_not_modified": True,
        },
        "warnings": warnings,
        "errors": errors,
    }
    if write_outputs:
        write_json(resolve(config["outputs"]["reports_dir"]) / "p11_fedtn_mps_verify_summary.json", summary)
    return summary
