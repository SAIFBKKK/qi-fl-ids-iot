"""Noise helpers for robustness experiments."""

from __future__ import annotations

import numpy as np


def gaussian_noise(shape: tuple[int, ...], *, std: float, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).normal(0.0, float(std), size=shape)
