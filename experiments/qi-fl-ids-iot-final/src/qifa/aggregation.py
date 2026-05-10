"""Parameter aggregation helpers for QIFA."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def parameter_drift(reference: list[np.ndarray], candidate: list[np.ndarray]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("parameter length mismatch while computing drift")
    squared_sum = 0.0
    count = 0
    for ref_tensor, cand_tensor in zip(reference, candidate):
        diff = np.asarray(cand_tensor, dtype=np.float64) - np.asarray(ref_tensor, dtype=np.float64)
        squared_sum += float(np.square(diff).sum())
        count += int(diff.size)
    return float(np.sqrt(squared_sum / max(count, 1)))


def aggregate_weighted_ndarrays(parameter_sets: Iterable[list[np.ndarray]], weights: list[float] | np.ndarray) -> list[np.ndarray]:
    arrays = list(parameter_sets)
    if not arrays:
        raise ValueError("cannot aggregate an empty parameter list")
    resolved_weights = np.asarray(weights, dtype=np.float64)
    if len(arrays) != int(resolved_weights.size):
        raise ValueError("parameter set count does not match weight count")
    first_shapes = [tensor.shape for tensor in arrays[0]]
    for params in arrays[1:]:
        if [tensor.shape for tensor in params] != first_shapes:
            raise ValueError("all client parameter tensors must share shapes")
    normalized = resolved_weights / max(float(resolved_weights.sum()), 1e-12)
    aggregated: list[np.ndarray] = []
    for tensor_index in range(len(arrays[0])):
        stacked = np.stack([np.asarray(params[tensor_index], dtype=np.float64) for params in arrays], axis=0)
        combined = np.tensordot(normalized, stacked, axes=(0, 0))
        aggregated.append(combined.astype(arrays[0][tensor_index].dtype, copy=False))
    return aggregated
