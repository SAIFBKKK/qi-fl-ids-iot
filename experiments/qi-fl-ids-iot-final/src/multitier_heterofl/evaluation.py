"""Evaluation utilities for P7 HeteroFL."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from fl_l1.client_data import make_dataloader as make_l1_dataloader
from fl_hierarchical.data import make_dataloader as make_l2_dataloader
from fl_hierarchical.metrics import classification_report, multiclass_metrics, one_vs_rest_rows, top_confusion_pairs
from models.metrics import binary_metrics, classification_report_dict


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    arrays: Any,
    task: str,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    make_loader = make_l1_dataloader if task == "l1_binary" else make_l2_dataloader
    loader = make_loader(arrays, batch_size=batch_size, shuffle=False, seed=seed, device=device)
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        losses.append(float(criterion(logits, y_batch).item()))
        y_true.append(y_batch.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.asarray([], dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.asarray([], dtype=np.int64)
    if task == "l1_binary":
        metrics = binary_metrics(true, pred)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["classification_report"] = classification_report_dict(true, pred)
        matrix = [
            [int(metrics["TN"]), int(metrics["FP"])],
            [int(metrics["FN"]), int(metrics["TP"])],
        ]
        one_vs_rest = [
            {
                "class_id": 0,
                "class_name": "normal",
                "TP": int(metrics["TN"]),
                "FP": int(metrics["FN"]),
                "TN": int(metrics["TP"]),
                "FN": int(metrics["FP"]),
                "precision": float(metrics["precision_normal"]),
                "recall": float(metrics["recall_normal"]),
                "f1": float(metrics["f1_normal"]),
            },
            {
                "class_id": 1,
                "class_name": "attack",
                "TP": int(metrics["TP"]),
                "FP": int(metrics["FP"]),
                "TN": int(metrics["TN"]),
                "FN": int(metrics["FN"]),
                "precision": float(metrics["precision_attack"]),
                "recall": float(metrics["recall_attack"]),
                "f1": float(metrics["f1_attack"]),
            },
        ]
    else:
        metrics = multiclass_metrics(true, pred, class_names)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["classification_report"] = classification_report(metrics)
        metrics["top_confusion_pairs"] = top_confusion_pairs(metrics["confusion_matrix"], class_names)
        matrix = metrics["confusion_matrix"]
        one_vs_rest = one_vs_rest_rows(metrics)
        metrics["FPR_macro"] = float(np.mean([row.get("FPR", 0.0) for row in metrics["per_class"].values()]))
        metrics["FNR_macro"] = float(np.mean([row.get("FNR", 0.0) for row in metrics["per_class"].values()]))
    metrics["confusion_matrix"] = matrix
    metrics["one_vs_rest_rows"] = one_vs_rest
    return {"metrics": metrics, "y_true": true, "y_pred": pred}
