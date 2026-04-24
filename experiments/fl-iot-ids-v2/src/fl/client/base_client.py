from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flwr.client import NumPyClient
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR
from src.data.datasets.flat_dataset import IoTLocalDataset
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


def compute_local_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from local client labels.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    nonzero = counts > 0
    weights = np.zeros_like(counts, dtype=np.float32)
    weights[nonzero] = counts.sum() / (num_classes * counts[nonzero])

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def compute_proximal_term(
    model: nn.Module,
    global_params: list[torch.Tensor],
) -> torch.Tensor:
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), global_params):
        proximal_term = proximal_term + torch.sum((local_param - global_param) ** 2)
    return proximal_term


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
        imbalance_strategy: str = "none",
        focal_gamma: float = 2.0,
        weight_decay: float = 0.0,
        proximal_mu: float = 0.0,
        fl_strategy: str = "fedavg",
    ) -> None:
        self.client_id = client_id
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.benign_class_id = benign_class_id
        self.rare_class_ids = rare_class_ids or DEFAULT_RARE_CLASS_IDS
        self.imbalance_strategy = imbalance_strategy
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        self.proximal_mu = proximal_mu
        self.fl_strategy = fl_strategy.lower()
        self.num_classes = output_dim

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

        if self.imbalance_strategy == "focal_loss":
            self.criterion = FocalLoss(gamma=self.focal_gamma)
            logger.info("Client %s using FocalLoss(gamma=%.1f)", self.client_id, self.focal_gamma)
        elif self.imbalance_strategy == "focal_loss_weighted":
            self.criterion = FocalLoss(gamma=self.focal_gamma, alpha=self._get_class_weights())
            logger.info("Client %s using FocalLoss(gamma=%.1f, alpha=weighted)", self.client_id, self.focal_gamma)
        elif self.imbalance_strategy == "class_weights":
            self.criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
            logger.info("Client %s using weighted CrossEntropyLoss", self.client_id)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self._reset_optimizer()

        logger.info(
            "Client %s loaded train=%s val=%s input_dim=%s output_dim=%s strategy=%s mu=%s",
            self.client_id,
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            input_dim,
            output_dim,
            self.fl_strategy,
            self.proximal_mu,
        )

    def _get_class_weights(self) -> torch.Tensor:
        """
        Priorité 1 : fichier global partagé (calculé sur le dataset complet).
        Priorité 2 : calcul local depuis les données du client (fallback).
        Le fichier global garantit des poids cohérents entre tous les clients.
        """
        global_weights_path = ARTIFACTS_DIR / "class_weights_34.pkl"
        if global_weights_path.exists():
            import pickle
            with open(global_weights_path, "rb") as f:
                weights = pickle.load(f)
            logger.info("Client %s: poids globaux chargés depuis %s", self.client_id, global_weights_path)
            if isinstance(weights, torch.Tensor):
                return weights.float().to(self.device)
            return torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Fallback : calcul local depuis les données du client
        logger.warning(
            "Client %s: %s introuvable — calcul local des poids (moins cohérent entre clients)",
            self.client_id,
            global_weights_path,
        )
        ds = self.train_loader.dataset
        y_train = (
            ds.dataset.y[ds.indices].cpu().numpy()
            if hasattr(ds, "dataset") and hasattr(ds, "indices")
            else ds.y.cpu().numpy()
        )
        return compute_local_class_weights(y_train, num_classes=self.num_classes).to(self.device)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def _reset_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self._reset_optimizer()
        self.model.train()

        global_params = [
            param.detach().clone().to(self.device)
            for param in self.model.parameters()
        ]

        use_fedprox = self.fl_strategy == "fedprox" and self.proximal_mu > 0.0

        start = now_perf()
        last_loss = 0.0
        last_ce_loss = 0.0
        last_prox_term = 0.0

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)

                ce_loss = self.criterion(logits, yb)

                if use_fedprox:
                    prox_term = compute_proximal_term(self.model, global_params)
                    loss = ce_loss + (self.proximal_mu / 2.0) * prox_term
                    last_prox_term = float(prox_term.item())
                else:
                    loss = ce_loss
                    last_prox_term = 0.0

                loss.backward()
                self.optimizer.step()

                last_ce_loss = float(ce_loss.item())
                last_loss = float(loss.item())

        updated_parameters = get_model_parameters(self.model)
        train_time_sec = elapsed_seconds(start)
        update_size_bytes = parameters_size_bytes(updated_parameters)

        metrics = {
            "client_id": self.client_id,
            "train_loss_last": float(last_loss),
            "train_ce_loss_last": float(last_ce_loss),
            "train_prox_term_last": float(last_prox_term),
            "proximal_mu": float(self.proximal_mu),
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
