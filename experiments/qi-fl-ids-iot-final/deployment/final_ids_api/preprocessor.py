from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model_loader import bundle_dir


class FeatureValidationError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class L1Preprocessor:
    def __init__(self, bundle: Path | None = None) -> None:
        self.bundle = bundle or bundle_dir()
        self.schema = _read_json(self.bundle / "feature_schema.json")
        self.selected_indices = [int(value) for value in self.schema.get("selected_indices", [])]
        self.input_dim = int(self.schema.get("selected_feature_count", 12))
        self.original_dim = int(self.schema.get("original_feature_count", 28))

    def transform_one(self, features: list[float]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 1:
            raise FeatureValidationError("features must be a flat numeric list")
        if len(values) == self.input_dim:
            return values.reshape(1, -1)
        if len(values) == self.original_dim and len(self.selected_indices) == self.input_dim:
            return values[self.selected_indices].reshape(1, -1)
        raise FeatureValidationError(f"expected {self.input_dim} selected features or {self.original_dim} original scaled features")

    def transform_batch(self, rows: list[list[float]]) -> np.ndarray:
        if not rows:
            raise FeatureValidationError("batch must contain at least one row")
        return np.concatenate([self.transform_one(row) for row in rows], axis=0)
