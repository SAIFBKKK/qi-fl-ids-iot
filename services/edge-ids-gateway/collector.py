from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable

from metrics import (
    EDGE_GATEWAY_ALERTS_TOTAL,
    EDGE_GATEWAY_ALLOWED_TOTAL,
    EDGE_GATEWAY_BLOCKED_TOTAL,
    EDGE_GATEWAY_MQTT_CONNECTED,
    EDGE_GATEWAY_MQTT_MESSAGES_TOTAL,
    EDGE_GATEWAY_REJECTED_TOTAL,
)


class MQTTEdgeGatewayCollector:
    def __init__(
        self,
        settings: Any,
        preprocessor: Any,
        inference_engine: Any,
        pipeline_runner: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.settings = settings
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.pipeline_runner = pipeline_runner
        self.started = False
        self.mqtt_connected = False
        self.mqtt = None
        self.client = None

    def start(self) -> None:
        if not self.settings.mqtt_enabled:
            return
        self.client = self._build_client()
        self.started = True
        self.client.connect_async(
            self.settings.mqtt_broker,
            self.settings.mqtt_port,
            keepalive=self.settings.mqtt_keepalive,
        )
        self.client.loop_start()

    def stop(self) -> None:
        if not self.settings.mqtt_enabled:
            self.started = False
            return
        if self.client is not None:
            self._publish_status("offline", mqtt_connected=False)
            self.client.loop_stop()
            self.client.disconnect()
        self.started = False

    def _build_client(self) -> Any:
        self._ensure_mqtt_module()
        try:
            client = self.mqtt.Client(
                self.mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.settings.mqtt_client_id,
            )
        except (AttributeError, TypeError):
            client = self.mqtt.Client(client_id=self.settings.mqtt_client_id)

        if self.settings.mqtt_username:
            client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password or None)

        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.will_set(
            self.settings.status_topic,
            self._json_payload(self._status_message("offline", mqtt_connected=False)),
            qos=self.settings.mqtt_qos,
            retain=True,
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        if self._reason_code_value(reason_code) != 0:
            self.mqtt_connected = False
            EDGE_GATEWAY_MQTT_CONNECTED.set(0)
            return

        self.mqtt_connected = True
        EDGE_GATEWAY_MQTT_CONNECTED.set(1)
        client.subscribe(self.settings.raw_input_topic, qos=self.settings.mqtt_qos)
        self._publish_status("online", mqtt_connected=True)

    def _on_disconnect(self, _client: Any, _userdata: Any, *args: Any) -> None:
        self.mqtt_connected = False
        EDGE_GATEWAY_MQTT_CONNECTED.set(0)

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        EDGE_GATEWAY_MQTT_MESSAGES_TOTAL.labels(topic=message.topic).inc()
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="invalid_json").inc()
            self._publish_blocked_invalid(
                reason="invalid_raw_schema",
                error=f"invalid JSON: {exc}",
                original_event=None,
            )
            return

        try:
            result = self.pipeline_runner(payload)
        except ValueError as exc:
            reason = self._classify_error(str(exc))
            EDGE_GATEWAY_REJECTED_TOTAL.labels(reason=reason).inc()
            self._publish_blocked_invalid(
                reason=self._blocked_reason(reason),
                error=str(exc),
                original_event=payload if isinstance(payload, dict) else None,
            )
            return
        except RuntimeError as exc:
            reason = self._classify_error(str(exc))
            EDGE_GATEWAY_REJECTED_TOTAL.labels(reason=reason).inc()
            self._publish_blocked_invalid(
                reason=self._blocked_reason(reason),
                error=str(exc),
                original_event=payload if isinstance(payload, dict) else None,
            )
            return
        except Exception as exc:  # noqa: BLE001
            EDGE_GATEWAY_REJECTED_TOTAL.labels(reason="inference").inc()
            self._publish_blocked_invalid(
                reason="inference",
                error=f"inference failed: {exc}",
                original_event=payload if isinstance(payload, dict) else None,
            )
            return

        prediction_message = {
            "schema_version": "1.0",
            "event_type": "edge_ids_prediction",
            "gateway_id": self.settings.gateway_id,
            "node_group": self.settings.node_group,
            "source_node_id": result["event"]["node_id"],
            "scenario": result["event"]["scenario"],
            "timestamp": result["event"]["timestamp"],
            "prediction": result["prediction"],
            "decision": result["decision"],
            "latency_ms": result["latency_ms"],
        }
        self._publish_json(self.settings.predictions_topic, prediction_message)

        if result["decision"] == "allow":
            EDGE_GATEWAY_ALLOWED_TOTAL.inc()
            self._publish_json(
                self.settings.accepted_topic,
                {
                    "schema_version": "1.0",
                    "event_type": "edge_ids_allowed",
                    "gateway_id": self.settings.gateway_id,
                    "node_group": self.settings.node_group,
                    "decision": "allow",
                    "original_event": result["event"],
                    "prediction": result["prediction"],
                    "latency_ms": result["latency_ms"],
                },
            )
        else:
            EDGE_GATEWAY_BLOCKED_TOTAL.inc()
            self._publish_json(
                self.settings.blocked_topic,
                {
                    "schema_version": "1.0",
                    "event_type": "edge_ids_blocked",
                    "gateway_id": self.settings.gateway_id,
                    "node_group": self.settings.node_group,
                    "decision": "block",
                    "reason": "ids_alert",
                    "original_event": result["event"],
                    "prediction": result["prediction"],
                    "latency_ms": result["latency_ms"],
                },
            )

        if result["prediction"]["is_alert"]:
            EDGE_GATEWAY_ALERTS_TOTAL.labels(severity=result["prediction"]["severity"]).inc()
            self._publish_json(
                self.settings.alerts_topic,
                {
                    "schema_version": "1.0",
                    "event_type": "edge_ids_alert",
                    "gateway_id": self.settings.gateway_id,
                    "node_group": self.settings.node_group,
                    "source_node_id": result["event"]["node_id"],
                    "scenario": result["event"]["scenario"],
                    "timestamp": result["event"]["timestamp"],
                    "alert": {
                        "label": result["prediction"]["predicted_label"],
                        "confidence": result["prediction"]["confidence"],
                        "severity": result["prediction"]["severity"],
                        "decision": result["decision"],
                    },
                },
            )

    def _publish_blocked_invalid(self, reason: str, error: str, original_event: dict[str, Any] | None) -> None:
        EDGE_GATEWAY_BLOCKED_TOTAL.inc()
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "event_type": "edge_ids_blocked",
            "gateway_id": self.settings.gateway_id,
            "node_group": self.settings.node_group,
            "decision": "block",
            "reason": reason,
            "error": error,
            "latency_ms": 0.0,
        }
        if original_event is not None:
            payload["original_event"] = original_event
        self._publish_json(self.settings.blocked_topic, payload)

    def _publish_status(self, status: str, mqtt_connected: bool) -> None:
        self.client.publish(
            self.settings.status_topic,
            self._json_payload(self._status_message(status, mqtt_connected=mqtt_connected)),
            qos=self.settings.mqtt_qos,
            retain=True,
        )

    def _status_message(self, status: str, mqtt_connected: bool) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_type": "edge_gateway_status",
            "service": "edge-ids-gateway",
            "gateway_id": self.settings.gateway_id,
            "node_group": self.settings.node_group,
            "status": status,
            "mqtt_connected": mqtt_connected,
            "model_ready": self.inference_engine is not None and self.preprocessor is not None,
            "timestamp": self._now(),
        }

    def _publish_json(self, topic: str, payload: dict[str, Any]) -> None:
        result = self.client.publish(
            topic,
            self._json_payload(payload),
            qos=self.settings.mqtt_qos,
        )
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"publish to {topic} failed with rc={result.rc}")

    @staticmethod
    def _json_payload(payload: dict[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":"))

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

    @staticmethod
    def _classify_error(message: str) -> str:
        lowered = message.lower()
        if "model not ready" in lowered:
            return "model_not_ready"
        if "mapped feature" in lowered or "feature vector" in lowered or "features must" in lowered:
            return "feature_mapping"
        if "ip" in lowered or "field '" in lowered or "raw event" in lowered or "schema" in lowered:
            return "raw_schema"
        return "inference"

    @staticmethod
    def _blocked_reason(reason: str) -> str:
        if reason == "raw_schema":
            return "invalid_raw_schema"
        if reason == "feature_mapping":
            return "feature_mapping"
        if reason == "model_not_ready":
            return "model_not_ready"
        return "inference"

    def _ensure_mqtt_module(self) -> None:
        if self.mqtt is not None:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is required when MQTT_ENABLED=true") from exc
        self.mqtt = mqtt
