"""Legacy Flower client for P6 hierarchical L2/L3 runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import flwr as fl

from fl_hierarchical.data import (
    client_id_from_cid,
    load_hierarchical_client_data,
    load_hierarchical_config,
    load_l2_index_scenario,
    load_task_spec,
)
from fl_hierarchical.models import build_model, get_parameters, set_parameters
from fl_hierarchical.runtime import configured_address, latest_run_id, prepare_run_paths
from fl_hierarchical.strategy import (
    client_metrics_payload,
    evaluate_arrays,
    select_device,
    train_local,
)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


class HierarchicalFlowerClient(fl.client.NumPyClient):
    """Real Flower NumPyClient over P3 L2 index-only partitions."""

    def __init__(
        self,
        *,
        client_id: str,
        config: dict[str, Any],
        repo_root: Path,
        scenario,
        task_spec,
        logs_dir: Path,
        max_samples_per_client: int | None,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.repo_root = repo_root
        self.scenario = scenario
        self.task_spec = task_spec
        self.training_cfg = config["training"]
        self.logs_dir = logs_dir
        self.device = select_device(str(self.training_cfg["device"]))
        _append_log(self.logs_dir / "flower_clients.log", f"{client_id} loading data | global_test_loaded=false")
        self.data = load_hierarchical_client_data(
            config,
            repo_root,
            scenario,
            task_spec,
            client_id=client_id,
            max_samples_per_client=max_samples_per_client,
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"{client_id} data loaded | task={task_spec.task} "
                f"train={self.data.train.num_samples}/{self.data.expected_train_samples} "
                f"val={self.data.val.num_samples}/{self.data.expected_val_samples} "
                "global_test_loaded=false"
            ),
        )
        self.model = build_model(task_spec.model_config, output_dim=task_spec.output_dim).to(self.device)
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"ClientApp ready | client_id={client_id} task={task_spec.task}",
        )

    def get_parameters(self, config: dict[str, Any]) -> list:
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters: list, config: dict[str, Any]) -> tuple[list, int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit started | round={round_number}")
        set_parameters(self.model, parameters)
        train_result = train_local(
            model=self.model,
            arrays=self.data.train,
            batch_size=int(self.training_cfg["batch_size"]),
            local_epochs=int(self.training_cfg["local_epochs"]),
            learning_rate=float(self.training_cfg["learning_rate"]),
            weight_decay=float(self.training_cfg["weight_decay"]),
            device=self.device,
            seed=int(self.training_cfg["seed"]) + round_number,
            num_classes=int(self.task_spec.output_dim),
            use_class_weights=bool(self.training_cfg.get("use_class_weights", True)),
        )
        updated = get_parameters(self.model)
        payload_size = sum(array.nbytes for array in updated)
        val_result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            class_names=self.task_spec.class_names,
        )
        metrics = client_metrics_payload(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=float(train_result["loss"]),
            val_result=val_result,
            fit_time_sec=float(train_result["fit_time_sec"]),
            payload_size=int(payload_size),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"{self.client_id} fit completed | round={round_number} "
                f"loss={metrics['local_train_loss']:.4f} macro_f1={metrics['local_macro_f1']:.4f}"
            ),
        )
        return updated, self.data.train.num_samples, metrics

    def evaluate(self, parameters: list, config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} evaluate started | round={round_number}")
        set_parameters(self.model, parameters)
        result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            class_names=self.task_spec.class_names,
        )
        metrics = client_metrics_payload(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=0.0,
            val_result=result,
            fit_time_sec=0.0,
            payload_size=sum(array.nbytes for array in parameters),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{self.client_id} evaluate completed | round={round_number} loss={metrics['local_val_loss']:.4f}",
        )
        return float(metrics["local_val_loss"]), self.data.val.num_samples, metrics


def start_legacy_client(
    *,
    config: dict[str, Any],
    repo_root: Path,
    task: str,
    client_id: str,
    alpha: float,
    clients: int,
    server_address: str,
    max_samples_per_client: int | None,
    run_id: str | None = None,
) -> None:
    task_spec = load_task_spec(config, repo_root, task)
    scenario = load_l2_index_scenario(config, repo_root, alpha=alpha, clients=clients)
    resolved_run_id = run_id or latest_run_id(
        config=config,
        repo_root=repo_root,
        task=task_spec.task,
        alpha=alpha,
        clients=clients,
    )
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        task=task_spec.task,
        alpha=alpha,
        clients=clients,
        run_id=resolved_run_id,
        mark_latest=False,
    )
    _append_log(
        run_paths.logs_dir / "flower_clients.log",
        f"{client_id} process starting | task={task_spec.task} address={server_address} run_id={resolved_run_id}",
    )
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connecting to server | address={server_address}")
    client = HierarchicalFlowerClient(
        client_id=client_id,
        config=config,
        repo_root=repo_root,
        scenario=scenario,
        task_spec=task_spec,
        logs_dir=run_paths.logs_dir,
        max_samples_per_client=max_samples_per_client,
    )
    try:
        fl.client.start_client(server_address=server_address, client=client.to_client())
        _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connection finished cleanly")
    except BaseException as exc:
        _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connection failed | error={exc}")
        raise


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one P6 hierarchical Flower client")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l2", "l3", "l2_family", "l3_attack_type"])
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--address", default=None)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    args = parse_args()
    config = load_hierarchical_config(args.config)
    address = configured_address(config, args.address or args.server_address)
    start_legacy_client(
        config=config,
        repo_root=Path.cwd().resolve(),
        task=args.task,
        client_id=args.client_id,
        alpha=float(args.alpha),
        clients=int(args.clients),
        server_address=address,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
