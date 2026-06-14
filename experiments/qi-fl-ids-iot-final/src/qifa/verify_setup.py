"""Readiness checks for P9 QIFA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl

from qifa.config import load_config, repo_path, write_json
from qifa.summary_schema import expected_artifacts, expected_figures


def verify_qifa_setup(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    repo_root = Path.cwd().resolve()
    qga_dir = repo_path(config, "inputs.qga_final_mask_dir")
    checks = {
        "p3_partitions_root_exists": repo_path(config, "inputs.partitions_root").exists(),
        "global_test_exists": repo_path(config, "inputs.global_test_npz").exists(),
        "p5_grid_summary_exists": repo_path(config, "inputs.p5_grid_summary").exists(),
        "p8_ablation_summary_exists": repo_path(config, "inputs.p8_ablation_summary").exists(),
        "qga_final_mask_ready": (qga_dir / "feature_mask.json").exists() and (qga_dir / "selection_decision.json").exists(),
        "flower_version_detected": bool(fl.__version__),
        "true_runtime_required": bool(config.get("flower", {}).get("true_runtime_required", False)),
        "client_count_matches_minimums": int(config["flower"]["min_fit_clients"]) == int(config["scenario"]["clients"]) == int(config["flower"]["min_available_clients"]),
    }
    accepted = all(bool(value) for value in checks.values())
    summary = {
        "accepted": accepted,
        "mode": "verify",
        "flower_version": fl.__version__,
        "architecture": "legacy Flower runtime with real server/client processes",
        "scenario": config["scenario"],
        "global_test_holdout": {
            "path": str(repo_path(config, "inputs.global_test_npz")),
            "test_sent_to_clients": False,
        },
        "checks": checks,
        "artifacts_expected": expected_artifacts(),
        "figures_expected": expected_figures(),
        "criteria": {
            "true_flower_runtime": True,
            "test_sent_to_clients_false": True,
            "qifa_weights_logged": True,
            "qga_mask_optional": True,
        },
        "warnings": [],
        "errors": [] if accepted else ["QIFA verify checks failed"],
    }
    if write_outputs:
        write_json(repo_path(config, "outputs.reports_dir") / "qifa_verify_summary.json", summary)
    return summary
