from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from src.common.config import get_project_root, load_yaml_config
from src.data.dataloader import create_dataloaders_for_node
from src.model.evaluate import evaluate_model
from src.model.network import MLPClassifier
from src.model.train import train_one_epoch

logger = logging.getLogger("run_client")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flower client for local V1")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--node-id", type=str, choices=["node1", "node2", "node3"], required=True)
    parser.add_argument("--server-address", type=str, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            key: torch.tensor(value, dtype=model.state_dict()[key].dtype)
            for key, value in params_dict
        }
    )
    model.load_state_dict(state_dict, strict=True)


class IoTFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module,
        train_loader: Any,
        eval_loader: Any,
        device: torch.device,
        local_epochs: int,
        learning_rate: float,
    ) -> None:
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        logging.getLogger("fl_client").info("[%s] get_parameters()", self.node_id)
        return get_parameters(self.model)

    def fit(self, parameters, config):
        flog = logging.getLogger("fl_client")
        flog.info("[%s] fit() started", self.node_id)

        set_parameters(self.model, parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        last_loss = None
        last_acc = None

        for _ in range(self.local_epochs):
            train_metrics = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
            )
            last_loss = float(train_metrics["loss"])
            last_acc = float(train_metrics["accuracy"])

        flog.info(
            "[%s] fit() done | loss=%.4f | acc=%.4f",
            self.node_id,
            last_loss,
            last_acc,
        )

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {"loss": last_loss, "accuracy": last_acc},
        )

    def evaluate(self, parameters, config):
        flog = logging.getLogger("fl_client")
        flog.info("[%s] evaluate() started", self.node_id)

        set_parameters(self.model, parameters)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate_model(
            model=self.model,
            loader=self.eval_loader,
            criterion=criterion,
            device=self.device,
        )

        loss = float(metrics["loss"])
        acc = float(metrics["accuracy"])

        flog.info(
            "[%s] evaluate() done | loss=%.4f | acc=%.4f",
            self.node_id,
            loss,
            acc,
        )

        return loss, len(self.eval_loader.dataset), {"accuracy": acc}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    cfg = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    client_cfg = cfg.get("client", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})

    node_id = args.node_id
    server_address = (
        args.server_address
        if args.server_address is not None
        else client_cfg.get("server_address", "127.0.0.1:8080")
    )
    local_epochs = (
        args.local_epochs
        if args.local_epochs is not None
        else client_cfg.get("local_epochs", 1)
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else client_cfg.get("batch_size", 256)
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else client_cfg.get("learning_rate", 0.001)
    )

    seed = int(runtime_cfg.get("seed", 42))
    device_name = runtime_cfg.get("device", "cpu")
    num_workers = int(runtime_cfg.get("num_workers", 0))

    set_seed(seed)

    project_root = get_project_root()
    processed_root = data_cfg.get("processed_dir", "data/processed")
    node_dir = project_root / processed_root / node_id

    if not node_dir.exists():
        raise FileNotFoundError(f"Processed node directory not found: {node_dir}")

    device = torch.device(device_name)

    train_loader, eval_loader = create_dataloaders_for_node(
        node_dir=node_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MLPClassifier(
        input_dim=int(model_cfg.get("input_dim", 33)),
        hidden_dims=model_cfg.get("hidden_dims", [128, 64]),
        num_classes=int(model_cfg.get("num_classes", 34)),
    )

    logger.info(
        "Starting local V1 Flower client | node_id=%s | server=%s | epochs=%s | batch_size=%s | lr=%s",
        node_id,
        server_address,
        local_epochs,
        batch_size,
        learning_rate,
    )
    logging.getLogger("fl_client").info(
        "[%s] ready | device=%s | samples=%s | batches=%s",
        node_id,
        device,
        len(train_loader.dataset),
        len(train_loader),
    )

    client = IoTFLClient(
        node_id=node_id,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
    )

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
