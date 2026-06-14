"""Evaluation and threshold utilities for P5 FedAvg L1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.metrics import (
    binary_metrics,
    classification_report_dict,
    optional_auc_metrics,
    predictions_from_threshold,
    select_thresholds,
    threshold_sweep,
)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    collect_probabilities: bool = False,
) -> dict[str, Any]:
    """Evaluate a binary model on one DataLoader."""

    model.eval()
    total_loss = 0.0
    total_rows = 0
    targets: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        prob_attack = torch.softmax(logits, dim=1)[:, 1]
        batch_size = int(y_batch.size(0))
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        targets.append(y_batch.cpu().numpy())
        probs.append(prob_attack.cpu().numpy())

    y_true = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    prob_attack_np = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    y_pred = predictions_from_threshold(prob_attack_np, threshold)
    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_rows, 1)
    metrics["threshold"] = float(threshold)
    result: dict[str, Any] = {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if collect_probabilities:
        result["prob_attack"] = prob_attack_np
    return result


def tune_threshold_on_validation(
    y_true: np.ndarray,
    prob_attack: np.ndarray,
    *,
    start: float,
    stop: float,
    step: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune attack threshold on validation only."""

    rows = threshold_sweep(y_true, prob_attack, start=start, stop=stop, step=step)
    return select_thresholds(rows), rows


def finalize_test_metrics(
    y_true: np.ndarray,
    prob_attack: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute full test metrics at the selected threshold."""

    y_pred = predictions_from_threshold(prob_attack, threshold)
    metrics = binary_metrics(y_true, y_pred)
    auc = optional_auc_metrics(y_true, prob_attack)
    metrics["roc_auc"] = auc["roc_auc"]
    metrics["pr_auc"] = auc["pr_auc"]
    metrics["roc_pr_warning"] = auc["warning"]
    metrics["classification_report"] = classification_report_dict(y_true, y_pred)
    return metrics
