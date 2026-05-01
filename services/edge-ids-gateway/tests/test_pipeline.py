"""Integration test: raw_iot_event -> validate -> map -> preprocess -> infer.

No mocks — uses the real deployment bundle end-to-end.
"""
from __future__ import annotations

import copy
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

from raw_schema import validate_raw_event  # noqa: E402
from feature_mapper import map_raw_to_features, CANONICAL_FEATURE_NAMES  # noqa: E402
from preprocessor import EdgeFeaturePreprocessor  # noqa: E402
from inference_api import EdgeInferenceEngine  # noqa: E402

RAW_EVENT = {
    "schema_version": "1.0",
    "event_type": "raw_iot_event",
    "timestamp": "2026-01-01T00:00:00Z",
    "node_id": "sensor-a1",
    "gateway_id": "node1",
    "node_group": "room-a",
    "device_type": "thermostat",
    "src_ip": "10.10.1.21",
    "dst_ip": "10.10.0.10",
    "src_port": 51544,
    "dst_port": 443,
    "protocol": "tcp",
    "packet_size": 820,
    "packet_count": 6,
    "duration_ms": 85,
    "bytes_in": 920,
    "bytes_out": 3980,
    "flags": {"syn": 1},
    "flag_counts": {},
    "scenario": "normal_traffic",
}


@pytest.fixture(scope="module")
def preprocessor():
    return EdgeFeaturePreprocessor(
        feature_names_path=str(BUNDLE / "feature_names.pkl"),
        scaler_path=str(BUNDLE / "scaler.pkl"),
    )


@pytest.fixture(scope="module")
def engine():
    return EdgeInferenceEngine(
        model_path=str(BUNDLE / "global_model.pth"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
        model_config_path=str(BUNDLE / "model_config.json"),
        threshold=0.5,
    )


def test_full_pipeline_end_to_end(preprocessor, engine):
    validated = validate_raw_event(copy.deepcopy(RAW_EVENT))
    features = map_raw_to_features(copy.deepcopy(RAW_EVENT))

    assert len(features) == 28
    assert list(features.keys()) == CANONICAL_FEATURE_NAMES

    scaled = preprocessor.transform(features)
    assert scaled.shape == (1, 28)
    assert np.isfinite(scaled).all()

    result = engine.predict(scaled)

    assert "predicted_label" in result
    assert "predicted_label_id" in result
    assert "confidence" in result
    assert "is_alert" in result
    assert isinstance(result["predicted_label"], str)
    assert isinstance(result["predicted_label_id"], int)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["is_alert"], bool)
    assert result["predicted_label"] in engine.id_to_label.values()
