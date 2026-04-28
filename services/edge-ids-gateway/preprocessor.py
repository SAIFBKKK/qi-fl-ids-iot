from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from feature_mapper import CANONICAL_FEATURE_NAMES, features_to_ordered_list


class EdgeFeaturePreprocessor:
    def __init__(self, feature_names_path: str, scaler_path: str):
        self.feature_names_path = Path(feature_names_path)
        self.scaler_path = Path(scaler_path)
        self.feature_names = self._load_feature_names()
        self.scaler = self._load_scaler()

    def transform(self, features: dict[str, float]) -> np.ndarray:
        self._validate_features(features)
        ordered_features = features_to_ordered_list(features)
        vector = np.asarray([ordered_features], dtype=np.float32)

        if vector.shape != (1, len(CANONICAL_FEATURE_NAMES)):
            raise ValueError(f"feature vector must have shape (1, {len(CANONICAL_FEATURE_NAMES)})")

        scaled = np.asarray(self.scaler.transform(vector), dtype=np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("scaled feature vector contains non-finite values")
        return scaled

    def _load_feature_names(self) -> list[str]:
        try:
            feature_names = list(joblib.load(self.feature_names_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load feature_names.pkl from {self.feature_names_path}: {exc}") from exc

        if len(feature_names) != len(CANONICAL_FEATURE_NAMES):
            raise ValueError(
                f"feature_names.pkl must contain {len(CANONICAL_FEATURE_NAMES)} features, found {len(feature_names)}"
            )
        if feature_names != CANONICAL_FEATURE_NAMES:
            raise ValueError("feature_names.pkl order does not match CANONICAL_FEATURE_NAMES")
        return feature_names

    def _load_scaler(self) -> Any:
        try:
            scaler = joblib.load(self.scaler_path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load scaler.pkl from {self.scaler_path}: {exc}") from exc

        scaler_features = getattr(scaler, "n_features_in_", len(CANONICAL_FEATURE_NAMES))
        if int(scaler_features) != len(CANONICAL_FEATURE_NAMES):
            raise ValueError(
                f"scaler expects {scaler_features} features, expected {len(CANONICAL_FEATURE_NAMES)}"
            )
        return scaler

    def _validate_features(self, features: dict[str, float]) -> None:
        if not isinstance(features, dict):
            raise ValueError("features must be a dictionary")

        received_feature_set = set(features.keys())
        expected_feature_set = set(CANONICAL_FEATURE_NAMES)
        missing = sorted(expected_feature_set - received_feature_set)
        unexpected = sorted(received_feature_set - expected_feature_set)
        if missing:
            raise ValueError(f"missing mapped feature '{missing[0]}'")
        if unexpected:
            raise ValueError(f"unexpected mapped feature '{unexpected[0]}'")

        for name in CANONICAL_FEATURE_NAMES:
            value = features[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"feature '{name}' must be numeric")
            numeric_value = float(value)
            if not isfinite(numeric_value):
                raise ValueError(f"feature '{name}' must be finite")
