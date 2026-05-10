"""True Flower runtime for P9 QIFA."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1_flower.task import client_fit_metrics, evaluate_arrays, parameter_payload_size, set_parameters, train_local
from qifa.config import load_config
from qifa.data import (
    assert_port_available,
    concatenate_validation_arrays,
    configured_address,
    latest_run_id,
    load_client_arrays,
    load_mask_info,
    load_scenario,
    make_run_id,
    prepare_run_paths,
)
from qifa.model import build_model, get_parameters, select_device
from qifa.strategy import QIFAStrategy


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


class QIFAFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        scenario,
        client_id: str,
        logs_dir: Path,
        max_samples_per_client: int | None,
        use_qga_mask: bool,
    ) -> None:
        self.config = config
        self.client_id = client_id
        self.logs_dir = logs_dir
        self.training_cfg = config["training"]
        self.device = select_device(str(self.training_cfg["device"]))
        self.mask_info = load_mask_info(config, use_qga_mask=use_qga_mask)
        _append_log(self.logs_dir / "flower_clients.log", f"{client_id} loading data | global_test_loaded=false")
        train, val = load_client_arrays(
            scenario,
            client_id=client_id,
            mask=self.mask_info["mask"],
            max_samples=max_samples_per_client,
            seed=int(self.training_cfg["seed"]),
        )
        self.train = train
        self.val = val
        self.model = build_model(config["model"]).to(self.device)
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"ClientApp ready | client_id={client_id} train={train.num_samples} val={val.num_samples} qga={use_qga_mask}",
        )

    def get_parameters(self, config: dict[str, Any]):
        return get_parameters(self.model)

    def fit(self, parameters, config: dict[str, Any]):
        round_number = int(config.get("server_round", 0))
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit started | round={round_number}")
        set_parameters(self.model, parameters)
        train_result = train_local(
            model=self.model,
            arrays=self.train,
            batch_size=int(self.training_cfg["batch_size"]),
            local_epochs=int(self.training_cfg["local_epochs"]),
            learning_rate=float(self.training_cfg["learning_rate"]),
            weight_decay=float(self.training_cfg["weight_decay"]),
            device=self.device,
            seed=int(self.training_cfg["seed"]) + round_number,
        )
        val_result = evaluate_arrays(
            model=self.model,
            arrays=self.val,
            batch_size=int(self.training_cfg["batch_size"]) * 2,
            device=self.device,
            seed=int(self.training_cfg["seed"]) + 50_000 + round_number,
            threshold=0.5,
            collect_probabilities=True,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.train,
            val_arrays=self.val,
            train_loss=float(train_result["loss"]),
            val_result=val_result,
            fit_time_sec=float(train_result["fit_time_sec"]),
            payload_size=parameter_payload_size(parameters),
        )
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit completed | round={round_number} loss={metrics['local_train_loss']:.4f}")
        return get_parameters(self.model), self.train.num_samples, metrics

    def evaluate(self, parameters, config: dict[str, Any]):
        round_number = int(config.get("server_round", 0))
        set_parameters(self.model, parameters)
        val_result = evaluate_arrays(
            model=self.model,
            arrays=self.val,
            batch_size=int(self.training_cfg["batch_size"]) * 2,
            device=self.device,
            seed=int(self.training_cfg["seed"]) + 60_000 + round_number,
            threshold=0.5,
            collect_probabilities=True,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.train,
            val_arrays=self.val,
            train_loss=float(val_result["metrics"]["loss"]),
            val_result=val_result,
            fit_time_sec=0.0,
            payload_size=parameter_payload_size(parameters),
        )
        return float(metrics["local_val_loss"]), self.val.num_samples, metrics


def build_qifa_strategy(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    variant: str,
    gamma: float,
    mode: str,
    runtime_mode: str,
    run_id: str | None,
    max_samples_per_client: int | None,
    use_qga_mask: bool,
):
    scenario = load_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        variant=variant,
        gamma=gamma,
        use_qga_mask=use_qga_mask,
        run_id=run_id,
    )
    mask_info = load_mask_info(config, use_qga_mask=use_qga_mask)
    runtime_config = dict(config)
    runtime_config["scenario"] = {"alpha": float(alpha), "clients": int(clients), "rounds": int(rounds)}
    runtime_config["model"] = dict(config["model"])
    runtime_config["model"]["input_dim"] = int(mask_info["selected_features_count"])
    validation_arrays = concatenate_validation_arrays(
        scenario,
        mask=mask_info["mask"],
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        seed=int(config["training"]["seed"]),
    )
    return QIFAStrategy(
        config=runtime_config,
        repo_root=repo_root,
        scenario=scenario,
        run_paths=run_paths,
        validation_arrays=validation_arrays,
        variant=variant,
        gamma=gamma,
        use_qga_mask=use_qga_mask,
        mask_info=mask_info,
        mode=mode,
        runtime_mode=runtime_mode,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
    )


def start_qifa_server(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    variant: str,
    gamma: float,
    address: str,
    mode: str,
    run_id: str | None = None,
    runtime_mode: str = "manual",
    max_samples_per_client: int | None = None,
    use_qga_mask: bool = False,
) -> dict[str, Any]:
    assert_port_available(address)
    strategy = build_qifa_strategy(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        rounds=rounds,
        variant=variant,
        gamma=gamma,
        mode=mode,
        runtime_mode=runtime_mode,
        run_id=run_id,
        max_samples_per_client=max_samples_per_client,
        use_qga_mask=use_qga_mask,
    )
    strategy._log_server(f"server starting | address={address} run_id={strategy.run_paths.run_id}")
    fl.server.start_server(server_address=address, config=fl.server.ServerConfig(num_rounds=int(rounds)), strategy=strategy)
    return strategy.finalize()


def start_qifa_client(
    *,
    config: dict[str, Any],
    repo_root: Path,
    client_id: str,
    alpha: float,
    clients: int,
    variant: str,
    gamma: float,
    address: str,
    mode: str,
    run_id: str | None = None,
    max_samples_per_client: int | None = None,
    use_qga_mask: bool = False,
) -> None:
    scenario = load_scenario(config, repo_root, alpha=alpha, clients=clients)
    resolved_run_id = run_id or latest_run_id(config=config, repo_root=repo_root, alpha=alpha, clients=clients, variant=variant, gamma=gamma, use_qga_mask=use_qga_mask)
    run_paths = prepare_run_paths(config=config, repo_root=repo_root, alpha=alpha, clients=clients, variant=variant, gamma=gamma, use_qga_mask=use_qga_mask, run_id=resolved_run_id, mark_latest=False)
    runtime_config = dict(config)
    runtime_config["model"] = dict(config["model"])
    runtime_config["model"]["input_dim"] = int(load_mask_info(config, use_qga_mask=use_qga_mask)["selected_features_count"])
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} process starting | address={address}")
    client = QIFAFlowerClient(
        config=runtime_config,
        scenario=scenario,
        client_id=client_id,
        logs_dir=run_paths.logs_dir,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        use_qga_mask=use_qga_mask,
    )
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connecting to server | address={address}")
    fl.client.start_client(server_address=address, client=client.to_client())


def run_qifa_smoke_subprocess(
    *,
    config_path: Path,
    alpha: float,
    clients: int,
    rounds: int,
    variant: str,
    gamma: float,
    address: str,
    max_samples_per_client: int,
    timeout_sec: int = 600,
    use_qga_mask: bool = False,
) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    assert_port_available(address)
    run_id = make_run_id()
    prepare_run_paths(config=config, repo_root=repo_root, alpha=alpha, clients=clients, variant=variant, gamma=gamma, use_qga_mask=use_qga_mask, run_id=run_id)
    scripts_dir = repo_root / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts"
    server_script = scripts_dir / "09_start_qifa_flower_server.py"
    client_script = scripts_dir / "09_start_qifa_flower_client.py"
    common = [
        "--config",
        str(config_path),
        "--alpha",
        str(alpha),
        "--clients",
        str(clients),
        "--variant",
        variant,
        "--gamma",
        str(gamma),
        "--address",
        address,
        "--run-id",
        run_id,
        "--mode",
        "smoke",
        "--max-samples-per-client",
        str(max_samples_per_client),
    ]
    if use_qga_mask:
        common.append("--use-qga-mask")
    server_cmd = [sys.executable, str(server_script), *common, "--rounds", str(rounds), "--runtime-label", "subprocess"]
    processes: list[tuple[str, subprocess.Popen]] = []
    handles = []
    run_paths = prepare_run_paths(config=config, repo_root=repo_root, alpha=alpha, clients=clients, variant=variant, gamma=gamma, use_qga_mask=use_qga_mask, run_id=run_id, mark_latest=False)

    def start(name: str, command: list[str]) -> None:
        log_path = run_paths.logs_dir / f"{name}_stdout.log"
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        processes.append((name, subprocess.Popen(command, cwd=str(repo_root), stdout=handle, stderr=subprocess.STDOUT, text=True)))

    try:
        start("server", server_cmd)
        time.sleep(5.0)
        for client_index in range(1, int(clients) + 1):
            client_cmd = [sys.executable, str(client_script), *common, "--client-id", f"client_{client_index}"]
            start(f"client_{client_index}", client_cmd)
        deadline = time.monotonic() + float(timeout_sec)
        for name, process in processes:
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        failures = [(name, process.returncode) for name, process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"QIFA subprocess failure(s): {failures}")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    summary_path = run_paths.artifacts_dir / "run_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))
