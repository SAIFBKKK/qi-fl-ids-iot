"""Tests for MQTTFlowCollector (iot-node) — no live broker required."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from metrics import NodeMetrics  # noqa: E402
import collector as collector_module  # noqa: E402
from collector import MQTTFlowCollector  # noqa: E402


@pytest.fixture(scope="module")
def preprocessor():
    return FlowPreprocessor(
        scaler_path=str(BUNDLE / "scaler.pkl"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
    )


@pytest.fixture(scope="module")
def inference_service():
    return InferenceService(
        model_path=str(BUNDLE / "global_model.pth"),
        label_mapping_path=str(BUNDLE / "label_mapping.json"),
    )


def _build_collector(preprocessor, inference_service):
    mock_client = MagicMock()
    publish_result = MagicMock()
    publish_result.rc = 0  # paho.mqtt.client.MQTT_ERR_SUCCESS
    mock_client.publish.return_value = publish_result

    metrics = NodeMetrics()

    with patch.object(collector_module.mqtt, "Client", return_value=mock_client):
        c = MQTTFlowCollector(
            node_id="node1",
            broker="localhost",
            port=1883,
            username=None,
            password=None,
            threshold=0.5,
            preprocessor=preprocessor,
            inference_service=inference_service,
            metrics=metrics,
        )
    return c, metrics


def _make_mqtt_message(topic: str, payload_bytes: bytes) -> MagicMock:
    msg = MagicMock()
    msg.topic = topic
    msg.payload = payload_bytes
    return msg


def test_valid_message_triggers_prediction_publish(preprocessor, inference_service):
    c, metrics = _build_collector(preprocessor, inference_service)
    features = {name: 1.0 for name in preprocessor.feature_names}
    payload = {
        "flow_id": "coll-test-001",
        "node_id": "node1",
        "timestamp": "2026-01-01T00:00:00Z",
        "features": features,
    }
    msg = _make_mqtt_message(c.flow_topic, json.dumps(payload).encode("utf-8"))

    c._on_message(None, None, msg)

    assert c.client.publish.called
    published_topics = [call.args[0] for call in c.client.publish.call_args_list]
    assert c.prediction_topic in published_topics


def test_invalid_message_triggers_rejection_counter(preprocessor, inference_service):
    c, metrics = _build_collector(preprocessor, inference_service)
    msg = _make_mqtt_message(c.flow_topic, b"{ not valid json }")

    before = sum(metrics.flows_rejected.values())
    c._on_message(None, None, msg)
    after = sum(metrics.flows_rejected.values())

    assert after > before
    assert "invalid_json" in metrics.flows_rejected
