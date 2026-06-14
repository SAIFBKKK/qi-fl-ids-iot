"""Flower ServerApp and simulation runner for P5.2 L1 FedAvg."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.server import ServerConfig
from flwr.simulation import run_simulation

from fl_l1_flower.client_app import create_client_app
from fl_l1_flower.data import (
    concatenate_validation_arrays,
    load_flower_config,
    load_scenario,
)
from fl_l1_flower.runtime import prepare_run_paths
from fl_l1_flower.strategy import FlowerL1FedAvgStrategy


def create_server_app(strategy: FlowerL1FedAvgStrategy, *, rounds: int) -> fl.server.ServerApp:
    """Build a Flower 1.8-compatible ServerApp."""

    return fl.server.ServerApp(
        config=ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )


def run_flower_l1_simulation(
    *,
    config_path: Path,
    alpha: float | None = None,
    clients: int | None = None,
    rounds: int | None = None,
    max_samples_per_client: int | None = None,
    mode: str = "smoke",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run a true Flower simulation for one P5.2 L1 scenario."""

    repo_root = Path.cwd().resolve()
    config = load_flower_config(config_path)
    resolved_alpha = float(alpha if alpha is not None else config["scenario"]["alpha"])
    resolved_clients = int(clients if clients is not None else config["scenario"]["clients"])
    resolved_rounds = int(rounds if rounds is not None else config["scenario"]["rounds"])
    config["scenario"]["alpha"] = resolved_alpha
    config["scenario"]["clients"] = resolved_clients
    config["scenario"]["rounds"] = resolved_rounds

    scenario = load_scenario(config, repo_root, alpha=resolved_alpha, clients=resolved_clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=resolved_alpha,
        clients=resolved_clients,
        run_id=run_id,
    )
    validation_arrays = concatenate_validation_arrays(
        scenario,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        seed=int(config["training"]["seed"]),
    )
    strategy = FlowerL1FedAvgStrategy(
        config=config,
        repo_root=repo_root,
        scenario=scenario,
        run_dir=run_paths.run_dir,
        validation_arrays=validation_arrays,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        mode=mode,
        run_id=run_paths.run_id,
        runtime_mode="simulation",
    )
    server_app = create_server_app(strategy, rounds=resolved_rounds)
    client_app = create_client_app(
        config=config,
        scenario=scenario,
        logs_dir=run_paths.logs_dir,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
    )
    backend_config: dict[str, Any] = {
        "init_args": {"include_dashboard": False},
        "client_resources": {"num_cpus": 1},
    }
    if os.name == "nt":
        backend_config["init_args"]["num_cpus"] = 1
    strategy.console.log(
        f"Launching Flower simulation runtime | alpha={resolved_alpha} K={resolved_clients} rounds={resolved_rounds}"
    )
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=resolved_clients,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=bool(config["flower"].get("stream_logs", True)),
    )
    return strategy.finalize()


app = fl.server.ServerApp()
