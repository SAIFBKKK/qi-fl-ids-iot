"""Fast MLP proxy evaluation used as QGA fitness."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from qga.data import L1Arrays, make_loader
from qga.feature_mask import apply_feature_mask
from qga.metrics import metrics_from_probabilities


class FastBinaryMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), 32),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate_mask_fast_mlp(
    mask: np.ndarray,
    train: L1Arrays,
    val: L1Arrays,
    *,
    seed: int,
    batch_size: int = 512,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
) -> dict[str, Any]:
    selected_count = int(np.asarray(mask).sum())
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    x_train = apply_feature_mask(train.X, mask)
    x_val = apply_feature_mask(val.X, mask)
    model = FastBinaryMLP(selected_count).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loader = make_loader(x_train, train.y, batch_size=batch_size, shuffle=True, seed=seed)
    for _ in range(int(epochs)):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(target_device)
            y_batch = y_batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    val_loader = make_loader(x_val, val.y, batch_size=batch_size, shuffle=False, seed=seed)
    model.eval()
    probs: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(target_device)
            y_batch = y_batch.to(target_device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            prob_attack = torch.softmax(logits, dim=1)[:, 1]
            probs.append(prob_attack.detach().cpu().numpy())
            total_loss += float(loss.item()) * int(y_batch.shape[0])
            total_rows += int(y_batch.shape[0])
    prob_np = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    metrics = metrics_from_probabilities(val.y, prob_np, threshold=0.5)
    metrics["loss"] = total_loss / max(total_rows, 1)
    metrics["features_count"] = selected_count
    return metrics
