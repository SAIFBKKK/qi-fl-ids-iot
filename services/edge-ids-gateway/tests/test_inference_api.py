"""Tests for EdgeInferenceEngine (edge-ids-gateway)."""
from __future__ import annotations

import json
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

from inference_api import EdgeInferenceEngine  # noqa: E402


@pytest.fixture(scope="module")
def engine() -> EdgeInferenceEngine:
    return EdgeInferenceEngine(
        model_path=str(BUNDLE / "global_model.pth"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
        model_config_path=str(BUNDLE / "model_config.json"),
        threshold=0.5,
    )


@pytest.fixture(scope="module")
def label_mapping_values(engine) -> set[str]:
    return set(engine.id_to_label.values())


@pytest.fixture(scope="module")
def scaled_vector() -> np.ndarray:
    return np.zeros((1, 28), dtype=np.float32)


def test_engine_loads_from_bundle(engine):
    assert engine is not None
    assert engine.input_dim == 28
    assert engine.num_classes == 34
    assert len(engine.id_to_label) == 34


def test_predict_returns_required_fields(engine, scaled_vector):
    result = engine.predict(scaled_vector)
    assert "predicted_label" in result
    assert "predicted_label_id" in result
    assert "confidence" in result
    assert "is_alert" in result


def test_confidence_is_between_0_and_1(engine, scaled_vector):
    result = engine.predict(scaled_vector)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predicted_label_is_in_label_mapping(engine, scaled_vector, label_mapping_values):
    result = engine.predict(scaled_vector)
    assert isinstance(result["predicted_label"], str)
    assert result["predicted_label"] in label_mapping_values
