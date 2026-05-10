"""Fast standalone evaluation for QGA L2 masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from fl_hierarchical.data import HierarchicalArrays, make_dataloader
from fl_hierarchical.models import build_model
from qga_l2.fitness_l2 import macro_metrics_from_confusion, multiclass_confusion


def _sample_arrays(arrays: HierarchicalArrays, *, max_samples: int | None, seed: int) -> HierarchicalArrays:
    if max_samples is None or arrays.num_samples <= int(max_samples):
        return arrays
    rng = np.random.default_rng(int(seed))
    # Stratified-ish cap by sampling uniformly from the already attack-only split.
    indices = np.sort(rng.choice(arrays.num_samples, size=int(max_samples), replace=False))
    return HierarchicalArrays(
        X=arrays.X[indices],
        y=arrays.y[indices],
        label_id_original=arrays.label_id_original[indices],
        row_id=arrays.row_id[indices],
    )


def train_fast_mlp_l2(
    train: HierarchicalArrays,
    val: HierarchicalArrays,
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
    max_samples: int | None,
    epochs: int = 2,
    batch_size: int = 512,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    train = _sample_arrays(train, max_samples=max_samples, seed=seed)
    val = _sample_arrays(val, max_samples=max_samples, seed=seed + 999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model({"input_dim": input_dim, "hidden_layers": [64], "dropout": 0.1}, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = make_dataloader(train, batch_size=batch_size, shuffle=True, seed=seed, device=device)
    for _ in range(int(epochs)):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        val_loader = make_dataloader(val, batch_size=batch_size, shuffle=False, seed=seed, device=device)
        for x_batch, _ in val_loader:
            logits = model(x_batch.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    matrix = multiclass_confusion(val.y, y_pred, output_dim)
    return macro_metrics_from_confusion(matrix)
