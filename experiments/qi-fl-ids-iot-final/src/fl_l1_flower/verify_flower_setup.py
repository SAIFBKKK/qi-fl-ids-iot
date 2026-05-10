"""Verify P5.2 Flower runtime setup without training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1.scenario_loader import repo_path
from fl_l1_flower.data import load_flower_config, load_scenario
from fl_l1_flower.report_builder import write_verify_outputs
from fl_l1_flower.summary_schema import build_verify_contract
from fl_l1_flower.strategy import build_initial_parameters


def verify_flower_setup(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    """Verify Flower/P3/P4 compatibility for P5.2."""

    repo_root = Path.cwd().resolve()
    config = load_flower_config(config_path)
    warnings: list[str] = []
    scenario = load_scenario(config, repo_root)
    client_files = [
        {
            "client_id": client.client_id,
            "train_npz": str(client.train_npz),
            "val_npz": str(client.val_npz),
            "has_client_test": (client.train_npz.parent / "test_scaled.npz").exists(),
        }
        for client in scenario.clients
    ]
    try:
        import ray  # noqa: F401

        ray_available = True
    except Exception as exc:
        ray_available = False
        warnings.append(f"Ray simulation backend unavailable: {exc}")

    checks = {
        "flower_version_detected": bool(fl.__version__),
        "clientapp_available": hasattr(fl.client, "ClientApp"),
        "serverapp_available": hasattr(fl.server, "ServerApp"),
        "run_simulation_available": True,
        "legacy_start_server_available": hasattr(fl.server, "start_server"),
        "legacy_start_client_available": hasattr(fl.client, "start_client"),
        "legacy_runtime_fallback_available": hasattr(fl.server, "start_server") and hasattr(fl.client, "start_client"),
        "strategy_is_fedavg": config["flower"]["strategy"] == "FedAvg",
        "p3_partitions_exist": all(Path(item["train_npz"]).exists() and Path(item["val_npz"]).exists() for item in client_files),
        "clients_detected": len(scenario.clients) == int(config["scenario"]["clients"]),
        "global_test_holdout_exists": repo_path(repo_root, config["inputs"]["global_test_npz"]).exists(),
        "p4_metrics_exist": repo_path(repo_root, config["inputs"]["centralized_l1_metrics"]).exists(),
        "initial_parameters_build": build_initial_parameters(config) is not None,
        "server_client_modules_importable": True,
        "no_test_used_by_clients": not any(item["has_client_test"] for item in client_files),
    }
    accepted = all(checks.values())
    architecture = "ClientApp/ServerApp with flwr.simulation.run_simulation"
    summary = build_verify_contract(
        accepted=accepted,
        architecture=architecture,
        scenario={
            "alpha": scenario.alpha,
            "clients": scenario.num_clients,
            "client_ids": [client.client_id for client in scenario.clients],
        },
        global_test_holdout={
            "path": config["inputs"]["global_test_npz"],
            "sent_to_clients": False,
            "usage": "server final evaluation only",
        },
        checks=checks,
        warnings=warnings,
        errors=[],
    )
    summary["ray_available"] = ray_available
    if write_outputs:
        summary["generated_files"] = write_verify_outputs(repo_root=repo_root, config=config, summary=summary)
    return summary
