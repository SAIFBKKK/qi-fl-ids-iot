from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


DEFAULT_BENIGN_CLASS_ID = 1
DEFAULT_RARE_CLASS_IDS = (0, 3, 30, 31, 33)
DEFAULT_NUM_CLASSES = 34


def _compute_benign_recall(y_true: np.ndarray, y_pred: np.ndarray, benign_class_id: int) -> float:
    benign_mask = y_true == benign_class_id
    if benign_mask.sum() == 0:
        return 0.0
    return float((y_pred[benign_mask] == benign_class_id).mean())


def _compute_rare_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_class_ids: tuple[int, ...] | list[int],
) -> float:
    recalls: list[float] = []
    for class_id in rare_class_ids:
        class_mask = y_true == class_id
        if class_mask.sum() == 0:
            continue
        recalls.append(float((y_pred[class_mask] == class_id).mean()))
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def evaluate_model(
    model,
    loader,
    criterion,
    device,
    benign_class_id: int = DEFAULT_BENIGN_CLASS_ID,
    rare_class_ids: tuple[int, ...] | list[int] = DEFAULT_RARE_CLASS_IDS,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    model.eval()

    running_loss = 0.0
    total = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total += y.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / max(len(loader), 1)

    if all_targets:
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision_macro = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        benign_recall = _compute_benign_recall(y_true, y_pred, benign_class_id)
        false_positive_rate = float(1.0 - benign_recall)
        rare_class_recall = _compute_rare_class_recall(y_true, y_pred, rare_class_ids)
        class_counts = _compute_class_counts(y_true, y_pred, num_classes)
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        precision_macro = 0.0
        recall_macro = 0.0
        benign_recall = 0.0
        false_positive_rate = 0.0
        rare_class_recall = 0.0
        class_counts = {}

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_macro": macro_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "benign_recall": benign_recall,
        "false_positive_rate": false_positive_rate,
        "rare_class_recall": rare_class_recall,
        "num_samples": total,
        **class_counts,
    }


def _compute_class_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    counts: dict[str, float] = {}
    for class_id in range(num_classes):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())
        counts[f"tp_class_{class_id}"] = float(tp)
        counts[f"fp_class_{class_id}"] = float(fp)
        counts[f"fn_class_{class_id}"] = float(fn)
    return counts
