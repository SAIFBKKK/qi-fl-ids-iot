"""Verify P6 hierarchical Flower setup without training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np

from fl_l1.scenario_loader import repo_path
from fl_hierarchical.data import (
    load_hierarchical_config,
    load_l2_index_scenario,
    load_task_spec,
)
from fl_hierarchical.report_builder import write_verify_outputs
from fl_hierarchical.summary_schema import build_verify_contract
from fl_hierarchical.strategy import build_initial_parameters


def verify_hierarchical_setup(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_hierarchical_config(config_path)
    warnings: list[str] = []
    errors: list[str] = []
    scenario = load_l2_index_scenario(config, repo_root)
    l2_spec = load_task_spec(config, repo_root, "l2")
    l3_spec = load_task_spec(config, repo_root, "l3")
    client_files = []
    for client in scenario.clients:
        client_files.append(
            {
                "client_id": client.client_id,
                "train_row_ids_npy": str(client.train_row_ids_npy),
                "val_row_ids_npy": str(client.val_row_ids_npy),
                "has_client_test": (client.train_row_ids_npy.parent / "test_scaled.npz").exists()
                or (client.train_row_ids_npy.parent / "test_row_ids.npy").exists(),
            }
        )
    try:
        l2_initial_parameters = build_initial_parameters(config, l2_spec)
        l3_initial_parameters = build_initial_parameters(config, l3_spec)
        initial_ok = bool(l2_initial_parameters.tensors and l3_initial_parameters.tensors)
    except Exception as exc:
        initial_ok = False
        errors.append(f"initial parameter build failed: {exc}")
    with np.load(repo_path(repo_root, config["inputs"]["l2_train_npz"]), allow_pickle=False) as npz:
        train_keys_ok = {"X", "y_family", "label_id_original", "row_id"}.issubset(set(npz.files))
        feature_count = int(npz["X"].shape[1])
    checks = {
        "flower_version_detected": bool(fl.__version__),
        "legacy_start_server_available": hasattr(fl.server, "start_server"),
        "legacy_start_client_available": hasattr(fl.client, "start_client"),
        "flower_runtime_available": hasattr(fl.server, "start_server") and hasattr(fl.client, "start_client"),
        "p2_l2_outputs_exist": all(
            repo_path(repo_root, config["inputs"][key]).exists()
            for key in ["l2_train_npz", "l2_val_npz", "l2_test_npz", "l2_manifest", "l2_family_mapping"]
        ),
        "p3_l2_index_only_partitions_exist": all(
            Path(item["train_row_ids_npy"]).exists() and Path(item["val_row_ids_npy"]).exists()
            for item in client_files
        ),
        "p3_storage_mode_index_only": scenario.manifest.get("storage_mode") == "index_only",
        "global_test_holdout_exists": scenario.global_test_npz.exists(),
        "global_test_holdout_protected": not any(item["has_client_test"] for item in client_files),
        "no_test_used_by_clients": not any(item["has_client_test"] for item in client_files),
        "l2_output_dim_is_8": l2_spec.output_dim == 8,
        "l3_output_dim_is_33": l3_spec.output_dim == 33,
        "feature_count_is_28": feature_count == 28,
        "npz_keys_ok": train_keys_ok,
        "initial_parameters_build": initial_ok,
        "deploy_l2_l3_false": bool(config.get("tasks", {}).get("deploy_l2_l3")) is False,
        "p4_l1_metrics_exist": repo_path(repo_root, config["inputs"]["p4_l1_metrics"]).exists(),
    }
    if not repo_path(repo_root, config["inputs"]["p5_grid_summary"]).exists():
        warnings.append("P5 grid summary not found; P6 can run, but L1/P5 contextual comparison will be lighter.")
    accepted = all(checks.values()) and not errors
    summary = build_verify_contract(
        accepted=accepted,
        scenario={
            "alpha": float(scenario.alpha),
            "clients": int(scenario.num_clients),
            "client_ids": [client.client_id for client in scenario.clients],
            "recommended_full_rounds": int(config["execution"]["full_rounds"]),
            "smoke_rounds": int(config["execution"]["smoke_rounds"]),
        },
        checks=checks,
        warnings=warnings,
        errors=errors,
    )
    summary["l2"] = {
        "output_dim": l2_spec.output_dim,
        "class_names": l2_spec.class_names,
        "architecture": "28 -> 128 -> 64 -> 8",
    }
    summary["l3"] = {
        "output_dim": l3_spec.output_dim,
        "class_names": l3_spec.class_names,
        "architecture": "28 -> 128 -> 64 -> 33",
    }
    summary["global_test_holdout"] = {
        "path": config["inputs"]["l2_test_npz"],
        "sent_to_clients": False,
        "usage": "server final evaluation only",
    }
    if write_outputs:
        summary["generated_files"] = write_verify_outputs(repo_root=repo_root, config=config, summary=summary)
    return summary
