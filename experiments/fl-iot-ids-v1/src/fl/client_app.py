from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from flwr.client import ClientApp
from flwr.common import Context

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR
from src.data.dataloader import create_dataloaders_for_node
from src.model.evaluate import evaluate_model
from src.model.network import MLPClassifier
from src.model.train import train_one_epoch
from src.model.losses import build_loss, load_class_weights


logger = get_logger("fl_client")


def get_model_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def resolve_node_id(context: Context) -> str:
    """Map simulation partition/node info to node1/node2/node3."""
    # Flower simulation usually provides a node/partition identity via context.
    # We normalize it to our local dataset naming.
    partition_id = None

    if hasattr(context, "node_id") and context.node_id is not None:
        partition_id = int(context.node_id)

    if partition_id is None:
        # fallback for environments where node_id isn't exposed the same way
        partition_id = 0

    # map 0,1,2 -> node1,node2,node3
    node_index = (partition_id % 3) + 1
    return f"node{node_index}"


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
    ):
        self.node_id = node_id
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_dir = DATA_DIR / "processed" / node_id
        self.train_loader, self.eval_loader = create_dataloaders_for_node(
            node_dir=node_dir,
            batch_size=batch_size,
            num_workers=0,
        )

        sample_batch = next(iter(self.train_loader))
        X_sample, _ = sample_batch
        input_dim = X_sample.shape[1]

        base_dataset = self.train_loader.dataset.dataset
        num_classes = int(base_dataset.y.max().item()) + 1

        self.model = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=(128, 64),
            dropout=0.2,
        ).to(self.device)

        class_weights_path = ARTIFACTS_DIR / "class_weights_34.pkl"
        class_weights = load_class_weights(class_weights_path, device=self.device)
        self.criterion = build_loss(class_weights=class_weights)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        logger.info(
            "[%s] ready | device=%s | samples=%s | batches=%s",
            self.node_id,
            self.device,
            len(self.train_loader.dataset),
            len(self.train_loader),
        )

    def get_parameters(self, config):
        logger.info("[%s] get_parameters()", self.node_id)
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        logger.info("[%s] fit() started", self.node_id)
        set_model_parameters(self.model, parameters)

        last = None
        for _ in range(self.local_epochs):
            last = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
            )

        logger.info(
            "[%s] fit() done | loss=%.4f | acc=%.4f",
            self.node_id,
            float(last["loss"]),
            float(last["accuracy"]),
        )

        return (
            get_model_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "loss": float(last["loss"]),
                "accuracy": float(last["accuracy"]),
            },
        )

    def evaluate(self, parameters, config):
        logger.info("[%s] evaluate() started", self.node_id)
        set_model_parameters(self.model, parameters)

        metrics = evaluate_model(
            model=self.model,
            loader=self.eval_loader,
            criterion=self.criterion,
            device=self.device,
        )

        logger.info(
            "[%s] evaluate() done | loss=%.4f | acc=%.4f",
            self.node_id,
            float(metrics["loss"]),
            float(metrics["accuracy"]),
        )

        return (
            float(metrics["loss"]),
            len(self.eval_loader.dataset),
            {"accuracy": float(metrics["accuracy"])},
        )


def client_fn(context: Context):
    run_config = context.run_config

    batch_size = int(run_config["batch-size"])
    local_epochs = int(run_config["local-epochs"])
    learning_rate = float(run_config["learning-rate"])

    node_id = resolve_node_id(context)

    logger.info("ClientApp started | resolved node_id=%s", node_id)

    return FlowerClient(
        node_id=node_id,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
    ).to_client()


app = ClientApp(client_fn=client_fn)
