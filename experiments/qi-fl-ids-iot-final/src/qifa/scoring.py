"""QIFA client scoring and weight transforms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_client_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return float(
        float(weights["macro_f1_weight"]) * float(metrics.get("local_macro_f1", 0.0))
        + float(weights["attack_recall_weight"]) * float(metrics.get("local_attack_recall", 0.0))
        - float(weights["fpr_penalty"]) * float(metrics.get("local_fpr", 0.0))
        - float(weights["loss_penalty"]) * float(metrics.get("local_val_loss", metrics.get("local_train_loss", 0.0)))
        - float(weights["drift_penalty"]) * float(metrics.get("drift", 0.0))
    )


def normalize_scores_to_theta(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    lower = float(values.min())
    upper = float(values.max())
    if math.isclose(lower, upper):
        return np.full(values.shape, math.pi / 4.0, dtype=float)
    normalized = (values - lower) / (upper - lower)
    eps = 1e-6
    return eps + normalized * ((math.pi / 2.0) - 2.0 * eps)


def amplitudes_from_theta(theta: list[float] | np.ndarray) -> np.ndarray:
    angles = np.asarray(theta, dtype=float)
    return np.sin(angles)


def probabilities_from_amplitudes(amplitudes: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(amplitudes, dtype=float)
    squared = np.square(values)
    total = float(squared.sum())
    if total <= 0.0:
        if squared.size == 0:
            return np.empty(0, dtype=float)
        return np.full(squared.shape, 1.0 / float(squared.size), dtype=float)
    return squared / total


def hybrid_weights(fedavg_weights: list[float] | np.ndarray, probabilities: list[float] | np.ndarray, gamma: float) -> np.ndarray:
    fedavg = np.asarray(fedavg_weights, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if fedavg.shape != probs.shape:
        raise ValueError(f"shape mismatch between fedavg_weights {fedavg.shape} and probabilities {probs.shape}")
    gamma_value = float(gamma)
    weights = (1.0 - gamma_value) * fedavg + gamma_value * probs
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=float)
    return weights / total


def shannon_entropy(probabilities: list[float] | np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.size == 0:
        return 0.0
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())
