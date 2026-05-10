"""Binary mask and theta-vector utilities for QGA."""

from __future__ import annotations

import numpy as np


def validate_mask(mask: np.ndarray, n_features: int) -> np.ndarray:
    array = np.asarray(mask, dtype=np.int8)
    if array.shape != (int(n_features),):
        raise ValueError(f"mask shape must be ({n_features},), got {array.shape}")
    if not np.isin(array, [0, 1]).all():
        raise ValueError("mask must be binary")
    return array


def random_mask(
    rng: np.random.Generator,
    *,
    n_features: int,
    min_features: int,
    max_features: int,
) -> np.ndarray:
    count = int(rng.integers(int(min_features), int(max_features) + 1))
    indices = rng.choice(int(n_features), size=count, replace=False)
    mask = np.zeros(int(n_features), dtype=np.int8)
    mask[indices] = 1
    return mask


def repair_mask_bounds(
    mask: np.ndarray,
    *,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    repaired = np.asarray(mask, dtype=np.int8).copy()
    count = int(repaired.sum())
    if count < int(min_features):
        off = np.flatnonzero(repaired == 0)
        add = rng.choice(off, size=int(min_features) - count, replace=False)
        repaired[add] = 1
    elif count > int(max_features):
        on = np.flatnonzero(repaired == 1)
        drop = rng.choice(on, size=count - int(max_features), replace=False)
        repaired[drop] = 0
    return repaired


def theta_to_probabilities(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.sin(theta) ** 2


def sample_quantum_mask(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    min_features: int,
    max_features: int,
) -> np.ndarray:
    probabilities = theta_to_probabilities(theta)
    mask = (rng.random(probabilities.shape[0]) < probabilities).astype(np.int8)
    return repair_mask_bounds(
        mask,
        min_features=min_features,
        max_features=max_features,
        rng=rng,
    )


def mutate_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mutation_rate: float,
    min_features: int,
    max_features: int,
) -> np.ndarray:
    mutated = np.asarray(mask, dtype=np.int8).copy()
    flips = rng.random(mutated.shape[0]) < float(mutation_rate)
    mutated[flips] = 1 - mutated[flips]
    return repair_mask_bounds(
        mutated,
        min_features=min_features,
        max_features=max_features,
        rng=rng,
    )


def update_theta_towards_best(
    theta: np.ndarray,
    best_mask: np.ndarray,
    *,
    update_rate: float = 0.12,
) -> np.ndarray:
    target = np.where(np.asarray(best_mask, dtype=np.int8) == 1, np.pi / 2, 0.0)
    updated = theta + float(update_rate) * (target - theta)
    return np.clip(updated, 0.0, np.pi / 2)
