"""Local client training logic for P5 FedAvg L1."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import nn

from .client_data import ClientArrays, binary_counts, make_dataloader
from .communication import model_size_bytes
from .evaluation import evaluate_loader


@dataclass(frozen=True)
class ClientFitResult:
    """Result sent from client to server after local training."""

    client_id: str
    state_dict: dict[str, torch.Tensor]
    num_examples: int
    metrics: dict[str, Any]
    upload_bytes: int
    download_bytes: int


class FedAvgL1Client:
    """In-process client used by P5 scripts and smoke runs."""

    def __init__(
        self,
        *,
        client_id: str,
        train_arrays: ClientArrays,
        val_arrays: ClientArrays,
        model_factory,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_epochs: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.client_id = client_id
        self.train_arrays = train_arrays
        self.val_arrays = val_arrays
        self.model_factory = model_factory
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.local_epochs = int(local_epochs)
        self.device = device
        self.seed = int(seed)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, global_state_dict: dict[str, torch.Tensor], *, round_number: int) -> ClientFitResult:
        """Train locally and return updated state."""

        model = self.model_factory().to(self.device)
        model.load_state_dict(global_state_dict, strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = make_dataloader(
            self.train_arrays,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed + round_number,
            device=self.device,
        )
        val_loader = make_dataloader(
            self.val_arrays,
            batch_size=self.batch_size * 4,
            shuffle=False,
            seed=self.seed,
            device=self.device,
        )

        start = perf_counter()
        last_loss = 0.0
        for _ in range(self.local_epochs):
            model.train()
            running_loss = 0.0
            batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batches += 1
            last_loss = running_loss / max(batches, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        fit_time_sec = perf_counter() - start

        eval_start = perf_counter()
        eval_result = evaluate_loader(
            model,
            val_loader,
            self.criterion,
            self.device,
            threshold=0.5,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        eval_time_sec = perf_counter() - eval_start

        updated_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        payload_size = model_size_bytes(updated_state)
        counts = binary_counts(self.train_arrays.y)
        metrics = {
            "round": int(round_number),
            "client_id": self.client_id,
            "train_samples": self.train_arrays.num_samples,
            "val_samples": self.val_arrays.num_samples,
            **counts,
            "local_train_loss": float(last_loss),
            "local_val_loss": float(eval_result["metrics"]["loss"]),
            "local_accuracy": float(eval_result["metrics"]["accuracy"]),
            "local_macro_f1": float(eval_result["metrics"]["macro_f1"]),
            "local_attack_recall": float(eval_result["metrics"]["recall_attack"]),
            "local_fpr": float(eval_result["metrics"]["FPR"]),
            "fit_time_sec": float(fit_time_sec),
            "eval_time_sec": float(eval_time_sec),
            "upload_bytes": int(payload_size),
            "download_bytes": int(payload_size),
        }
        return ClientFitResult(
            client_id=self.client_id,
            state_dict=updated_state,
            num_examples=self.train_arrays.num_samples,
            metrics=metrics,
            upload_bytes=payload_size,
            download_bytes=payload_size,
        )
