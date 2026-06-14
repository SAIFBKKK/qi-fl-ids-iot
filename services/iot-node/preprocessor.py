from __future__ import annotations

import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class ProcessedFlow:
    flow_id: str
    node_id: str | None
    source_timestamp: str | None
    raw_features: dict[str, float]
    feature_vector: np.ndarray
    scaled_vector: np.ndarray


class FlowSchemaError(ValueError):
    def __init__(self, reason: str, message: str, flow_id: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.flow_id = flow_id


class FlowPreprocessor:
    def __init__(self, scaler_path: str, label_mapping_path: str) -> None:
        self.scaler_path = Path(scaler_path)
        self.bundle_path = Path(label_mapping_path).parent
        self.feature_names_path = self.bundle_path / "feature_names.pkl"
        self.feature_names = self._load_required_joblib(self.feature_names_path, "feature_names")
        self.scaler = self._load_required_joblib(self.scaler_path, "scaler")
        self.expected_feature_set = set(self.feature_names)

        if len(self.feature_names) != 28:
            self._fail_startup(f"feature_names.pkl must contain 28 features, found {len(self.feature_names)}")

        logger.info("preprocessor_ready", feature_count=len(self.feature_names), scaler_path=str(self.scaler_path))

    def preprocess(self, payload: dict[str, Any]) -> ProcessedFlow:
        flow_id = self._require_string(payload, "flow_id")
        features = payload.get("features")
        if not isinstance(features, dict):
            raise FlowSchemaError("invalid_schema", "flow message must contain a JSON object at 'features'", flow_id)

        self._validate_feature_schema(flow_id, features)
        raw_features = self._coerce_features(flow_id, features)
        feature_vector = np.asarray([[raw_features[name] for name in self.feature_names]], dtype=np.float32)

        try:
            scaled_vector = np.asarray(self.scaler.transform(feature_vector), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - runtime preprocessing failure rejects only this flow.
            raise FlowSchemaError("preprocessing_error", f"scaler transform failed: {exc}", flow_id) from exc

        return ProcessedFlow(
            flow_id=flow_id,
            node_id=self._optional_string(payload, "node_id"),
            source_timestamp=self._optional_string(payload, "timestamp"),
            raw_features=raw_features,
            feature_vector=feature_vector,
            scaled_vector=scaled_vector,
        )

    def _validate_feature_schema(self, flow_id: str, features: dict[str, Any]) -> None:
        received_feature_set = set(features.keys())
        missing = sorted(self.expected_feature_set - received_feature_set)
        unexpected = sorted(received_feature_set - self.expected_feature_set)

        if missing:
            logger.warning("flow_id={} rejected: missing feature '{}'", flow_id, missing[0])
            raise FlowSchemaError("missing_feature", f"missing feature '{missing[0]}'", flow_id)

        if unexpected:
            logger.warning("flow_id={} rejected: unexpected feature '{}'", flow_id, unexpected[0])
            raise FlowSchemaError("unexpected_feature", f"unexpected feature '{unexpected[0]}'", flow_id)

    def _coerce_features(self, flow_id: str, features: dict[str, Any]) -> dict[str, float]:
        safe_features: dict[str, float] = {}
        for name in self.feature_names:
            try:
                numeric_value = float(features[name])
            except (TypeError, ValueError) as exc:
                logger.warning("flow_id={} rejected: invalid numeric value for feature '{}'", flow_id, name)
                raise FlowSchemaError("invalid_feature_value", f"invalid numeric value for feature '{name}'", flow_id) from exc

            if not isfinite(numeric_value):
                logger.warning("flow_id={} rejected: non-finite value for feature '{}'", flow_id, name)
                raise FlowSchemaError("invalid_feature_value", f"non-finite value for feature '{name}'", flow_id)

            safe_features[name] = numeric_value
        return safe_features

    def _load_required_joblib(self, path: Path, name: str) -> Any:
        try:
            value = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - bundle integrity is mandatory in P2.
            self._fail_startup(f"Cannot load {name} from {path}: {exc}")
        return value

    @staticmethod
    def _require_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise FlowSchemaError("invalid_schema", f"flow message must contain a non-empty string at '{key}'")
        return value

    @staticmethod
    def _optional_string(payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _fail_startup(message: str) -> None:
        logger.critical("Cannot load model bundle: {}", message)
        logger.critical("Service refuses to start - fix the bundle first")
        sys.exit(1)
