from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.common.logger import get_logger
from src.data.datasets.flat_dataset import IoTLocalDataset, load_npz_arrays
from src.fl.metrics.classification import compute_classification_metrics
from src.fl.metrics.convergence import elapsed_seconds, now_perf, parameters_size_bytes
from src.fl.metrics.rare_attack import compute_benign_metrics, compute_rare_class_recall
from src.models.flat.network import FlatMLP

logger = get_logger(__name__)

DEFAULT_BENIGN_CLASS_ID = 1
DEFAULT_RARE_CLASS_IDS = [0, 3, 30, 31, 33]


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError("Parameter length mismatch")

    updated_state = {
        k: torch.tensor(v, dtype=state_dict[k].dtype)
        for k, v in zip(keys, parameters)
    }
    model.load_state_dict(updated_state, strict=True)


class BaseIDSClient(NumPyClient):
    def __init__(
        self,
        client_id: str,
        train_path: str | Path,
        val_path: str | Path | None,
        input_dim: int = 28,
        output_dim: int = 34,
        batch_size: int = 64,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
        benign_class_id: int = DEFAULT_BENIGN_CLASS_ID,
        rare_class_ids: list[int] | None = None,
        split_seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.benign_class_id = benign_class_id
        self.rare_class_ids = rare_class_ids or DEFAULT_RARE_CLASS_IDS

        train_dataset = IoTLocalDataset(train_path)

        if val_path is not None and Path(val_path).exists():
            val_dataset = IoTLocalDataset(val_path)
        else:
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size

            generator = torch.Generator().manual_seed(split_seed)
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=generator,
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model = FlatMLP(input_dim=input_dim, output_dim=output_dim)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(
            "Client %s loaded train=%s val=%s input_dim=%s output_dim=%s",
            self.client_id,
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            input_dim,
            output_dim,
        )

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.train()

        start = now_perf()
        last_loss = 0.0

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())

        updated_parameters = get_model_parameters(self.model)
        train_time_sec = elapsed_seconds(start)
        update_size_bytes = parameters_size_bytes(updated_parameters)

        metrics = {
            "client_id": self.client_id,
            "train_loss_last": float(last_loss),
            "train_time_sec": float(train_time_sec),
            "update_size_bytes": int(update_size_bytes),
            "num_examples": int(len(self.train_loader.dataset)),
        }

        return (
            updated_parameters,
            len(self.train_loader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.eval()

        start = now_perf()

        total_loss = 0.0
        total_examples = 0
        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                preds = torch.argmax(logits, dim=1)

                total_loss += float(loss.item()) * xb.size(0)
                total_examples += int(xb.size(0))
                all_preds.append(preds.cpu().numpy())
                all_true.append(yb.cpu().numpy())

        eval_time_sec = elapsed_seconds(start)

        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)

        avg_loss = total_loss / max(total_examples, 1)

        cls_metrics = compute_classification_metrics(y_true, y_pred)
        benign_metrics = compute_benign_metrics(
            y_true, y_pred, benign_class_id=self.benign_class_id
        )
        rare_recall = compute_rare_class_recall(
            y_true, y_pred, rare_class_ids=self.rare_class_ids
        )

        metrics = {
            "client_id": self.client_id,
            "eval_loss": float(avg_loss),
            "eval_time_sec": float(eval_time_sec),
            "num_examples": int(total_examples),
            **cls_metrics,
            **benign_metrics,
            "rare_class_recall": float(rare_recall),
        }

        return float(avg_loss), total_examples, metrics