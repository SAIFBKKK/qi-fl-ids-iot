from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import paho.mqtt.client as mqtt
from loguru import logger

from inference_api import InferenceService, PredictionResult
from metrics import NodeMetrics
from preprocessor import FlowPreprocessor, FlowSchemaError


class MQTTFlowCollector:
    def __init__(
        self,
        node_id: str,
        broker: str,
        port: int,
        username: str | None,
        password: str | None,
        threshold: float,
        preprocessor: FlowPreprocessor,
        inference_service: InferenceService,
        metrics: NodeMetrics,
    ) -> None:
        self.node_id = node_id
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.threshold = threshold
        self.preprocessor = preprocessor
        self.inference_service = inference_service
        self.metrics = metrics
        self.flow_topic = f"ids/flows/{node_id}"
        self.prediction_topic = f"ids/predictions/{node_id}"
        self.alert_topic = f"ids/alerts/{node_id}"
        self.status_topic = f"ids/status/{node_id}"
        self.client = self._build_client()

    def start(self) -> None:
        logger.info("mqtt_collector_starting", broker=self.broker, port=self.port, node_id=self.node_id)
        self.client.connect_async(self.broker, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        logger.info("mqtt_collector_stopping", node_id=self.node_id)
        self._publish_status("offline")
        self.client.loop_stop()
        self.client.disconnect()

    def _build_client(self) -> mqtt.Client:
        client_id = f"iot-node-{self.node_id}"
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)

        if self.username:
            client.username_pw_set(self.username, self.password)

        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.will_set(self.status_topic, self._status_payload("offline"), qos=1, retain=True)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _on_connect(self, client: mqtt.Client, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        if self._reason_code_value(reason_code) != 0:
            self.metrics.mark_mqtt_connected(False)
            self.metrics.mark_error(f"mqtt connect failed: {reason_code}")
            logger.error("mqtt_connect_failed", reason_code=str(reason_code), broker=self.broker, port=self.port)
            return

        self.metrics.mark_mqtt_connected(True)
        client.subscribe(self.flow_topic, qos=1)
        self._publish_status("online")
        logger.info("mqtt_connected", broker=self.broker, port=self.port, subscribed_topic=self.flow_topic)

    def _on_disconnect(self, _client: mqtt.Client, _userdata: Any, *args: Any) -> None:
        self.metrics.mark_mqtt_connected(False)
        reason_code = args[1] if len(args) >= 2 else args[0] if args else "unknown"
        logger.warning("mqtt_disconnected", node_id=self.node_id, reason_code=str(reason_code))

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        self.metrics.mark_received(message.topic)
        started_at = time.perf_counter()

        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.metrics.mark_rejected("invalid_json", str(exc))
            logger.warning("flow_message_rejected", topic=message.topic, reason="invalid_json", error=str(exc))
            return

        try:
            if not isinstance(payload, dict):
                raise FlowSchemaError("invalid_schema", "MQTT payload must be a JSON object")

            processed_flow = self.preprocessor.preprocess(payload)
            if processed_flow.node_id and processed_flow.node_id != self.node_id:
                raise FlowSchemaError(
                    "invalid_schema",
                    f"payload node_id={processed_flow.node_id} does not match {self.node_id}",
                    processed_flow.flow_id,
                )

            prediction = self.inference_service.predict(processed_flow)
            self.metrics.observe_inference_latency(time.perf_counter() - started_at)

            is_alert = prediction.confidence >= self.threshold and not is_benign_label(prediction.predicted_label)
            prediction_message = self._prediction_message(processed_flow.flow_id, prediction, is_alert)
            self._publish_json(self.prediction_topic, prediction_message)
            self.metrics.mark_prediction(prediction.predicted_label)

            if is_alert:
                severity = severity_for_confidence(prediction.confidence)
                alert_message = self._alert_message(processed_flow.flow_id, prediction, message.topic, severity)
                self._publish_json(self.alert_topic, alert_message)
                self.metrics.mark_alert(severity, prediction.predicted_label)

            logger.info(
                "flow_processed",
                flow_id=processed_flow.flow_id,
                predicted_label=prediction.predicted_label,
                confidence=prediction.confidence,
                is_alert=is_alert,
            )
        except FlowSchemaError as exc:
            self.metrics.mark_rejected(exc.reason, str(exc))
            logger.warning(
                "flow_message_rejected",
                topic=message.topic,
                flow_id=exc.flow_id,
                reason=exc.reason,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - MQTT callback must not crash network loop.
            self.metrics.mark_rejected("inference_error", str(exc))
            logger.exception("flow_processing_failed", topic=message.topic, error=str(exc))

    def _prediction_message(
        self,
        flow_id: str,
        prediction: PredictionResult,
        is_alert: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_type": "ids_prediction",
            "node_id": self.node_id,
            "timestamp": self._now(),
            "flow_id": flow_id,
            "predicted_label": prediction.predicted_label,
            "predicted_label_id": prediction.predicted_label_id,
            "confidence": prediction.confidence,
            "is_alert": is_alert,
            "model_version": prediction.model_version,
        }

    def _alert_message(
        self,
        flow_id: str,
        prediction: PredictionResult,
        source_topic: str,
        severity: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_type": "ids_alert",
            "node_id": self.node_id,
            "timestamp": self._now(),
            "flow_id": flow_id,
            "predicted_label": prediction.predicted_label,
            "predicted_label_id": prediction.predicted_label_id,
            "confidence": prediction.confidence,
            "severity": severity,
            "source_topic": source_topic,
            "model_version": prediction.model_version,
        }

    def _publish_status(self, status: str) -> None:
        self.client.publish(self.status_topic, self._status_payload(status), qos=1, retain=True)

    def _status_payload(self, status: str) -> str:
        payload = {
            "schema_version": "1.0",
            "event_type": "ids_status",
            "node_id": self.node_id,
            "timestamp": self._now(),
            "status": status,
            "model_version": "baseline_fedavg_normal_classweights",
        }
        return json.dumps(payload, separators=(",", ":"))

    def _publish_json(self, topic: str, payload: dict[str, Any]) -> None:
        result = self.client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"publish to {topic} failed with rc={result.rc}")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _reason_code_value(reason_code: Any) -> int:
        value = getattr(reason_code, "value", reason_code)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0 if str(value).lower() == "success" else 1


def is_benign_label(label: str) -> bool:
    return "benign" in label.lower()


def severity_for_confidence(confidence: float) -> str:
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"
