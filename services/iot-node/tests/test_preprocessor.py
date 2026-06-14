"""Tests for FlowPreprocessor (iot-node)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

BUNDLE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments"
    / "fl-iot-ids-v3"
    / "outputs"
    / "deployment"
    / "baseline_fedavg_normal_classweights"
)

from preprocessor import FlowPreprocessor, FlowSchemaError  # noqa: E402


@pytest.fixture(scope="module")
def preprocessor():
    return FlowPreprocessor(
        scaler_path=str(BUNDLE / "scaler.pkl"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
    )


@pytest.fixture(scope="module")
def feature_names(preprocessor):
    return preprocessor.feature_names


def _valid_payload(feature_names: list[str]) -> dict:
    return {
        "flow_id": "flow-test-001",
        "node_id": "node1",
        "timestamp": "2026-01-01T00:00:00Z",
        "features": {name: 1.0 for name in feature_names},
    }


def test_valid_payload_passes(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    result = preprocessor.preprocess(payload)
    assert result.flow_id == "flow-test-001"
    assert len(result.raw_features) == 28
    assert result.feature_vector.shape == (1, 28)
    assert result.scaled_vector.shape == (1, 28)


def test_missing_feature_raises(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    del payload["features"][feature_names[0]]
    with pytest.raises(FlowSchemaError) as exc_info:
        preprocessor.preprocess(payload)
    assert exc_info.value.reason == "missing_feature"


def test_unexpected_extra_feature_raises(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    payload["features"]["__unexpected_extra__"] = 0.0
    with pytest.raises(FlowSchemaError) as exc_info:
        preprocessor.preprocess(payload)
    assert exc_info.value.reason == "unexpected_feature"


def test_nan_value_raises(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    payload["features"][feature_names[0]] = float("nan")
    with pytest.raises(FlowSchemaError) as exc_info:
        preprocessor.preprocess(payload)
    assert exc_info.value.reason == "invalid_feature_value"


def test_non_numeric_value_raises(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    payload["features"][feature_names[0]] = "not_a_number"
    with pytest.raises(FlowSchemaError) as exc_info:
        preprocessor.preprocess(payload)
    assert exc_info.value.reason == "invalid_feature_value"


def test_feature_order_preserved(preprocessor, feature_names):
    payload = _valid_payload(feature_names)
    for i, name in enumerate(feature_names):
        payload["features"][name] = float(i + 1)
    result = preprocessor.preprocess(payload)
    for i, name in enumerate(feature_names):
        assert result.raw_features[name] == float(i + 1)
    expected = np.array([[float(i + 1) for i in range(28)]], dtype=np.float32)
    np.testing.assert_array_equal(result.feature_vector, expected)
