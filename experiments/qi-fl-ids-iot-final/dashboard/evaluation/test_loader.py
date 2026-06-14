from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifact_resolver import resolve


def load_test_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    target = resolve(path)
    with np.load(target, allow_pickle=False) as data:
        return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y_binary"], dtype=np.int64)


def load_qga_indices(mask_path: str | Path) -> list[int]:
    payload = json.loads(resolve(mask_path).read_text(encoding="utf-8"))
    if "selected_indices" in payload:
        return [int(index) for index in payload["selected_indices"]]
    mask = payload.get("mask", payload.get("feature_mask", []))
    return [idx for idx, value in enumerate(mask) if int(value) == 1]


def apply_mask(X: np.ndarray, indices: list[int] | None) -> np.ndarray:
    if not indices:
        return X
    return X[:, indices]
