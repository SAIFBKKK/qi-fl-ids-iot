"""Tests for InferenceService (iot-node)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BUNDLE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments"
    / "fl-iot-ids-v3"
    / "outputs"
    / "deployment"
    / "baseline_fedavg_normal_classweights"
)

from preprocessor import FlowPreprocessor  # noqa: E402
from inference_api import InferenceService  # noqa: E402


@pytest.fixture(scope="module")
def label_mapping() -> dict[int, str]:
    with open(BUNDLE / "label_mapping.json", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["id_to_label"].items()}


@pytest.fixture(scope="module")
def inference_service() -> InferenceService:
    return InferenceService(
        model_path=str(BUNDLE / "global_model.pth"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
    )


@pytest.fixture(scope="module")
def preprocessor() -> FlowPreprocessor:
    return FlowPreprocessor(
        scaler_path=str(BUNDLE / "scaler.pkl"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
    )


@pytest.fixture(scope="module")
def processed_flow(preprocessor):
    feature_names = preprocessor.feature_names
    payload = {
        "flow_id": "inf-test-001",
        "node_id": "node1",
        "timestamp": "2026-01-01T00:00:00Z",
        "features": {name: 1.0 for name in feature_names},
    }
    return preprocessor.preprocess(payload)


def test_inference_service_loads(inference_service):
    assert inference_service is not None
    assert inference_service.engine_type == "torch_mlp"


def test_predict_returns_label_in_mapping(inference_service, processed_flow, label_mapping):
    result = inference_service.predict(processed_flow)
    assert result.predicted_label in label_mapping.values()


def test_confidence_is_float_between_0_and_1(inference_service, processed_flow):
    result = inference_service.predict(processed_flow)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_predicted_label_id_is_integer_in_mapping(inference_service, processed_flow, label_mapping):
    result = inference_service.predict(processed_flow)
    assert isinstance(result.predicted_label_id, int)
    assert result.predicted_label_id in label_mapping
