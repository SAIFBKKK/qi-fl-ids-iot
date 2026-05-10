"""PyTorch task utilities for the P5.2 Flower L1 runtime."""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from fl_l1.client_data import ClientArrays, binary_counts, make_dataloader
from fl_l1.evaluation import evaluate_loader
from fl_l1_flower.communication import model_size_bytes
from models.l1_mlp import CentralizedL1MLP


def select_device(device_config: str) -> torch.device:
    """Resolve the configured Torch device."""

    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def build_model(model_cfg: dict[str, Any]) -> CentralizedL1MLP:
    """Build the L1 MLP used by P4/P5/P5.2."""

    return CentralizedL1MLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_layers=list(model_cfg["hidden_layers"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
    )


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Convert Torch state_dict to Flower NDArrays."""

    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load Flower NDArrays into a Torch model."""

    state_keys = list(model.state_dict().keys())
    if len(state_keys) != len(parameters):
        raise ValueError(f"parameter length mismatch: expected {len(state_keys)}, got {len(parameters)}")
    state_dict = OrderedDict(
        (key, torch.tensor(array, dtype=model.state_dict()[key].dtype))
        for key, array in zip(state_keys, parameters)
    )
    model.load_state_dict(state_dict, strict=True)


def train_local(
    *,
    model: nn.Module,
    arrays: ClientArrays,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train one client locally."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = make_dataloader(arrays, batch_size=batch_size, shuffle=True, seed=seed, device=device)
    start = perf_counter()
    last_loss = 0.0
    for epoch in range(int(local_epochs)):
        model.train()
        running_loss = 0.0
        batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
        last_loss = running_loss / max(batches, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"loss": float(last_loss), "fit_time_sec": float(perf_counter() - start)}


def evaluate_arrays(
    *,
    model: nn.Module,
    arrays: ClientArrays,
    batch_size: int,
    device: torch.device,
    seed: int,
    threshold: float = 0.5,
    collect_probabilities: bool = False,
) -> dict[str, Any]:
    """Evaluate one arrays object with the shared P5 L1 metric code."""

    criterion = nn.CrossEntropyLoss()
    loader = make_dataloader(
        arrays,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    start = perf_counter()
    result = evaluate_loader(
        model,
        loader,
        criterion,
        device,
        threshold=threshold,
        collect_probabilities=collect_probabilities,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["eval_time_sec"] = float(perf_counter() - start)
    return result


def client_fit_metrics(
    *,
    client_id: str,
    round_number: int,
    train_arrays: ClientArrays,
    val_arrays: ClientArrays,
    train_loss: float,
    val_result: dict[str, Any],
    fit_time_sec: float,
    payload_size: int,
) -> dict[str, Any]:
    """Build the Flower client metrics payload."""

    val_metrics = val_result["metrics"]
    return {
        "round": int(round_number),
        "server_round": int(round_number),
        "client_id": client_id,
        "train_samples": train_arrays.num_samples,
        "val_samples": val_arrays.num_samples,
        **binary_counts(train_arrays.y),
        "local_train_loss": float(train_loss),
        "local_val_loss": float(val_metrics["loss"]),
        "local_accuracy": float(val_metrics["accuracy"]),
        "local_macro_f1": float(val_metrics["macro_f1"]),
        "local_attack_recall": float(val_metrics["recall_attack"]),
        "local_fpr": float(val_metrics["FPR"]),
        "fit_time_sec": float(fit_time_sec),
        "eval_time_sec": float(val_result["eval_time_sec"]),
        "upload_bytes": int(payload_size),
        "download_bytes": int(payload_size),
        "TP": int(val_metrics["TP"]),
        "TN": int(val_metrics["TN"]),
        "FP": int(val_metrics["FP"]),
        "FN": int(val_metrics["FN"]),
    }


def parameter_payload_size(parameters: list[np.ndarray] | dict[str, torch.Tensor]) -> int:
    """Return tensor payload size in bytes."""

    return model_size_bytes(parameters)

