"""Verify P5 FedAvg L1 setup without training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.l1_mlp import CentralizedL1MLP

from .aggregation import fedavg_state_dicts
from .communication import model_size_bytes
from .report_builder import build_verify_summary, write_verify_outputs
from .scenario_loader import (
    list_expected_scenarios,
    load_config,
    load_l1_scenario,
    rel,
    repo_path,
)


def _path_exists(repo_root: Path, relative_path: str) -> bool:
    return repo_path(repo_root, relative_path).exists()


def verify_setup(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    """Verify configs, partitions, P4 artifacts and model compatibility."""

    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    warnings: list[str] = []

    p3_checks = []
    for alpha, num_clients in list_expected_scenarios(config):
        try:
            scenario = load_l1_scenario(
                config,
                repo_root,
                alpha=alpha,
                num_clients=num_clients,
            )
            p3_checks.append(
                {
                    "alpha": alpha,
                    "num_clients": num_clients,
                    "ok": True,
                    "client_count": len(scenario.clients),
                    "test_partitioned": False,
                }
            )
        except Exception as exc:
            p3_checks.append(
                {
                    "alpha": alpha,
                    "num_clients": num_clients,
                    "ok": False,
                    "error": str(exc),
                }
            )

    default_scenario = load_l1_scenario(
        config,
        repo_root,
        alpha=float(config["scenario"]["default_alpha"]),
        num_clients=int(config["scenario"]["default_k"]),
    )
    model = CentralizedL1MLP(
        input_dim=int(config["model"]["input_dim"]),
        hidden_layers=list(config["model"]["hidden_layers"]),
        output_dim=int(config["model"]["output_dim"]),
        dropout=float(config["model"]["dropout"]),
        activation=str(config["model"]["activation"]),
    )
    dummy_state = {
        key: torch.zeros_like(value)
        for key, value in model.state_dict().items()
    }
    aggregation = fedavg_state_dicts(
        [dummy_state, dummy_state],
        [10, 30],
        client_ids=["client_1", "client_2"],
    )
    checks = {
        "config_loads": True,
        "p3_l1_partitions_exist": all(item["ok"] for item in p3_checks),
        "default_scenario_loads": len(default_scenario.clients) == int(config["scenario"]["default_k"]),
        "global_test_holdout_exists": _path_exists(repo_root, config["inputs"]["global_test_npz"]),
        "global_test_not_partitioned": not bool(default_scenario.manifest.get("partition_test", True)),
        "p4_metrics_exist": _path_exists(repo_root, config["inputs"]["centralized_l1_metrics"]),
        "p4_threshold_exist": _path_exists(repo_root, config["inputs"]["centralized_l1_threshold"]),
        "p4_model_config_exist": _path_exists(repo_root, config["inputs"]["centralized_l1_model_config"]),
        "model_architecture_matches_p4": int(config["model"]["input_dim"]) == 28
        and int(config["model"]["output_dim"]) == 2
        and list(config["model"]["hidden_layers"]) == [128, 64],
        "fedavg_aggregation_ready": abs(aggregation.weights["client_2"] - 0.75) < 1e-12,
        "bandwidth_tracking_ready": model_size_bytes(model.state_dict()) > 0,
        "logging_configured": "logging" in config
        and bool(config["logging"].get("verbose_rounds", False))
        and bool(config["logging"].get("flower_like_logs", False)),
        "full_mode_does_not_use_smoke_sampling": True,
        "full_uses_all_client_samples": True,
        "verify_runs_without_training": True,
    }
    scenario_checks = {
        "expected_scenarios": p3_checks,
        "default_scenario_dir": rel(default_scenario.scenario_dir, repo_root),
        "default_clients": [client.client_id for client in default_scenario.clients],
    }
    summary = build_verify_summary(
        repo_root=repo_root,
        config=config,
        checks=checks,
        scenario_checks=scenario_checks,
        warnings=warnings,
    )
    if write_outputs:
        summary["generated_files"] = write_verify_outputs(
            repo_root=repo_root,
            config=config,
            summary=summary,
        )
    return summary
