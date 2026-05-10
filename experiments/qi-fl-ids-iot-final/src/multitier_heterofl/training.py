"""Local training for P7 HeteroFL clients."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from torch import nn

from fl_l1.client_data import make_dataloader as make_l1_dataloader
from fl_hierarchical.data import make_dataloader as make_l2_dataloader


def select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def train_local(
    *,
    model: nn.Module,
    arrays: Any,
    task: str,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    make_loader = make_l1_dataloader if task == "l1_binary" else make_l2_dataloader
    loader = make_loader(arrays, batch_size=batch_size, shuffle=True, seed=seed, device=device)
    start = perf_counter()
    last_loss = 0.0
    for _ in range(int(local_epochs)):
        model.train()
        running = 0.0
        batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        last_loss = running / max(batches, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"loss": float(last_loss), "fit_time_sec": float(perf_counter() - start)}
