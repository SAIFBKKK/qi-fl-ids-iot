"""Flower ClientApp for P5.2 L1 FedAvg."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1_flower.data import (
    client_id_from_cid,
    load_flower_client_data,
)
from fl_l1_flower.task import (
    build_model,
    client_fit_metrics,
    evaluate_arrays,
    get_parameters,
    parameter_payload_size,
    select_device,
    set_parameters,
    train_local,
)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


class FlowerL1Client(fl.client.NumPyClient):
    """Real Flower NumPyClient over final P3 L1 partitions."""

    def __init__(
        self,
        *,
        client_id: str,
        config: dict[str, Any],
        scenario,
        logs_dir: Path,
        max_samples_per_client: int | None,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.training_cfg = config["training"]
        self.logs_dir = logs_dir
        self.device = select_device(str(self.training_cfg["device"]))
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{client_id} loading data | global_test_loaded=false",
        )
        self.data = load_flower_client_data(
            scenario,
            client_id=client_id,
            max_samples=max_samples_per_client,
            seed=int(self.training_cfg["seed"]),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"{client_id} data loaded | train={self.data.train.num_samples} "
                f"val={self.data.val.num_samples} global_test_loaded=false"
            ),
        )
        self.model = build_model(config["model"]).to(self.device)
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"ClientApp ready | client_id={client_id} "
                f"train={self.data.train.num_samples}/{self.data.expected_train_samples} "
                f"val={self.data.val.num_samples}/{self.data.expected_val_samples}"
            ),
        )

    def get_parameters(self, config: dict[str, Any]) -> list:
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters: list, config: dict[str, Any]) -> tuple[list, int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{self.client_id} fit started | round={round_number}",
        )
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
        )
        updated = get_parameters(self.model)
        payload_size = parameter_payload_size(updated)
        val_result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=0.5,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=float(train_result["loss"]),
            val_result=val_result,
            fit_time_sec=float(train_result["fit_time_sec"]),
            payload_size=payload_size,
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"{self.client_id} fit completed | round={round_number} "
                f"loss={metrics['local_train_loss']:.4f} "
                f"macro_f1={metrics['local_macro_f1']:.4f}"
            ),
        )
        return updated, self.data.train.num_samples, metrics

    def evaluate(self, parameters: list, config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{self.client_id} evaluate started | round={round_number}",
        )
        set_parameters(self.model, parameters)
        result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=0.5,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=0.0,
            val_result=result,
            fit_time_sec=0.0,
            payload_size=parameter_payload_size(parameters),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"{self.client_id} evaluate completed | round={round_number} "
                f"loss={metrics['local_val_loss']:.4f} macro_f1={metrics['local_macro_f1']:.4f}"
            ),
        )
        return float(metrics["local_val_loss"]), self.data.val.num_samples, metrics


def make_client_fn(
    *,
    config: dict[str, Any],
    scenario,
    logs_dir: Path,
    max_samples_per_client: int | None,
):
    """Create a Flower 1.8-compatible client function."""

    def client_fn(cid: str):
        client_id = client_id_from_cid(str(cid), scenario.num_clients)
        return FlowerL1Client(
            client_id=client_id,
            config=config,
            scenario=scenario,
            logs_dir=logs_dir,
            max_samples_per_client=max_samples_per_client,
        ).to_client()

    return client_fn


def create_client_app(
    *,
    config: dict[str, Any],
    scenario,
    logs_dir: Path,
    max_samples_per_client: int | None,
) -> fl.client.ClientApp:
    """Build the Flower ClientApp."""

    return fl.client.ClientApp(
        client_fn=make_client_fn(
            config=config,
            scenario=scenario,
            logs_dir=logs_dir,
            max_samples_per_client=max_samples_per_client,
        )
    )
