"""Controlled poisoning transforms for P10.

All functions operate on copies and never write back to P3 partition files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoisoningResult:
    X: np.ndarray
    y: np.ndarray
    manifest: dict[str, Any]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sample_indices(indices: np.ndarray, rate: float, seed: int) -> np.ndarray:
    if rate <= 0.0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)
    count = int(np.floor(float(rate) * int(indices.size)))
    count = max(0, min(count, int(indices.size)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return _rng(seed).choice(indices, size=count, replace=False).astype(np.int64)


def apply_label_flip(X: np.ndarray, y: np.ndarray, poison_rate: float, seed: int) -> PoisoningResult:
    poisoned_y = np.asarray(y).copy()
    candidates = np.arange(poisoned_y.shape[0], dtype=np.int64)
    selected = _sample_indices(candidates, poison_rate, seed)
    poisoned_y[selected] = 1 - poisoned_y[selected]
    return PoisoningResult(
        X=np.asarray(X).copy(),
        y=poisoned_y,
        manifest={
            "attack_type": "label_flip",
            "poison_rate": float(poison_rate),
            "eligible_rows": int(candidates.size),
            "poisoned_rows": int(selected.size),
            "labels_flipped": int(selected.size),
        },
    )


def apply_attack_to_normal(X: np.ndarray, y: np.ndarray, poison_rate: float, seed: int) -> PoisoningResult:
    poisoned_y = np.asarray(y).copy()
    attack_indices = np.flatnonzero(poisoned_y == 1).astype(np.int64)
    selected = _sample_indices(attack_indices, poison_rate, seed)
    poisoned_y[selected] = 0
    return PoisoningResult(
        X=np.asarray(X).copy(),
        y=poisoned_y,
        manifest={
            "attack_type": "attack_to_normal",
            "poison_rate": float(poison_rate),
            "eligible_rows": int(attack_indices.size),
            "poisoned_rows": int(selected.size),
            "attack_to_normal_rows": int(selected.size),
        },
    )


def apply_feature_noise(
    X: np.ndarray,
    y: np.ndarray,
    poison_rate: float,
    seed: int,
    *,
    std: float = 0.05,
    clip_min: float | None = -10.0,
    clip_max: float | None = 10.0,
) -> PoisoningResult:
    poisoned_X = np.asarray(X).copy()
    clean_y = np.asarray(y).copy()
    candidates = np.arange(clean_y.shape[0], dtype=np.int64)
    selected = _sample_indices(candidates, poison_rate, seed)
    if selected.size:
        noise = _rng(seed + 17).normal(loc=0.0, scale=float(std), size=poisoned_X[selected].shape)
        poisoned_X[selected] = poisoned_X[selected] + noise.astype(poisoned_X.dtype, copy=False)
        if clip_min is not None or clip_max is not None:
            poisoned_X[selected] = np.clip(poisoned_X[selected], clip_min, clip_max)
    return PoisoningResult(
        X=poisoned_X,
        y=clean_y,
        manifest={
            "attack_type": "feature_noise",
            "poison_rate": float(poison_rate),
            "eligible_rows": int(candidates.size),
            "poisoned_rows": int(selected.size),
            "noise_std": float(std),
            "labels_changed": 0,
        },
    )


def apply_poisoning(
    X: np.ndarray,
    y: np.ndarray,
    *,
    attack_type: str,
    poison_rate: float,
    seed: int,
    noise_std: float = 0.05,
    clip_min: float | None = -10.0,
    clip_max: float | None = 10.0,
) -> PoisoningResult:
    if attack_type == "clean" or float(poison_rate) <= 0.0:
        return PoisoningResult(
            X=np.asarray(X).copy(),
            y=np.asarray(y).copy(),
            manifest={
                "attack_type": attack_type,
                "poison_rate": float(poison_rate),
                "eligible_rows": int(len(y)),
                "poisoned_rows": 0,
            },
        )
    if attack_type == "label_flip":
        return apply_label_flip(X, y, poison_rate, seed)
    if attack_type == "attack_to_normal":
        return apply_attack_to_normal(X, y, poison_rate, seed)
    if attack_type == "feature_noise":
        return apply_feature_noise(X, y, poison_rate, seed, std=noise_std, clip_min=clip_min, clip_max=clip_max)
    raise ValueError(f"unsupported attack_type: {attack_type}")
