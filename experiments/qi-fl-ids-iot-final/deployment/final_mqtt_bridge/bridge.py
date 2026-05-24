from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from metrics import BridgeMetrics


MODEL_VERSION = "P8_FedAvg_QGA"
SELECTED_MASK_ID = "conservative_seed_42"
SOURCE = "final-mqtt-bridge"


@dataclass(frozen=True)
class BridgeSettings:
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str | None
    mqtt_password: str | None
    mqtt_client_id: str
    mqtt_qos: int
    subscribe_topic: str
    final_ids_api_url: str
    bridge_node_id: str
    feature_schema_path: Path
    http_timeout_sec: float

    @classmethod
    def from_env(cls) -> "BridgeSettings":
        return cls(
            mqtt_host=os.getenv("MQTT_HOST") or os.getenv("MQTT_BROKER", "mosquitto"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_username=os.getenv("MQTT_USERNAME", "ids_user"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
            mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "final-mqtt-bridge"),
            mqtt_qos=int(os.getenv("MQTT_QOS", "1")),
            subscribe_topic=os.getenv("MQTT_SUBSCRIBE_TOPIC", "ids/flows/#"),
            final_ids_api_url=os.getenv("FINAL_IDS_API_URL", "http://final-ids-api:8014"),
            bridge_node_id=os.getenv("BRIDGE_NODE_ID", "final-mqtt-bridge"),
            feature_schema_path=Path(os.getenv("FEATURE_SCHEMA_PATH", "/app/l1_final/feature_schema.json")),
            http_timeout_sec=float(os.getenv("HTTP_TIMEOUT_SEC", "5.0")),
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def node_id_from_topic(topic: str) -> str | None:
    parts = topic.split("/")
    if len(parts) >= 3 and parts[0] == "ids" and parts[1] == "flows":
        return parts[2] or None
    return None


def severity_for_attack(confidence: float) -> str:
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


class FeatureAdapter:
    def __init__(self, feature_schema_path: Path) -> None:
        self.feature_schema_path = feature_schema_path
        self.schema = self._load_schema(feature_schema_path)
        self.all_features = [str(item) for item in self.schema.get("all_features", [])]
        self.selected_features = [str(item) for item in self.schema.get("selected_features", [])]
        self.selected_indices = [int(item) for item in self.schema.get("selected_indices", [])]
        self.selected_count = int(self.schema.get("selected_feature_count", len(self.selected_features) or 12))
        self.original_count = int(self.schema.get("original_feature_count", len(self.all_features) or 28))

    @staticmethod
    def _load_schema(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def vector_from_payload(self, payload: dict[str, Any]) -> list[float]:
        features = payload.get("features")
        if features is None:
            raise ValueError("missing features")

        if isinstance(features, list):
            vector = [self._float(value, f"features[{index}]") for index, value in enumerate(features)]
            if len(vector) not in {self.selected_count, self.original_count}:
                raise ValueError(
                    f"feature list length {len(vector)} is not {self.selected_count} selected or {self.original_count} original features"
                )
            return vector

        if not isinstance(features, dict):
            raise ValueError("features must be a list or object")

        if self.all_features and all(name in features for name in self.all_features):
            return [self._float(features[name], name) for name in self.all_features]

        if self.selected_features and all(name in features for name in self.selected_features):
            return [self._float(features[name], name) for name in self.selected_features]

        missing_original = [name for name in self.all_features if name not in features][:5]
        missing_selected = [name for name in self.selected_features if name not in features][:5]
        raise ValueError(
            "feature object does not contain the required selected or original feature set; "
            f"missing_original={missing_original}, missing_selected={missing_selected}"
        )

    @staticmethod
    def _float(value: Any, name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature {name!r} is not numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"feature {name!r} is not finite")
        return numeric


class FinalMQTTBridge:
    def __init__(self, settings: BridgeSettings, metrics: BridgeMetrics) -> None:
        self.settings = settings
        self.metrics = metrics
        self.feature_adapter = FeatureAdapter(settings.feature_schema_path)
        self.http = httpx.Client(base_url=settings.final_ids_api_url.rstrip("/"), timeout=settings.http_timeout_sec)
        self.mqtt = None
        self.client = None

    def start(self) -> None:
        self._ensure_mqtt()
        try:
            self.client = self.mqtt.Client(
                self.mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.settings.mqtt_client_id,
            )
        except (AttributeError, TypeError):
            self.client = self.mqtt.Client(client_id=self.settings.mqtt_client_id)

        if self.settings.mqtt_username:
            self.client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password)

        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.connect_async(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
        self.http.close()
        self.metrics.set_ready(False)
        self.metrics.set_mqtt_connected(False)

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        if self._reason_code_value(reason_code) != 0:
            self.metrics.set_mqtt_connected(False)
            self.metrics.set_ready(False)
            self.metrics.mark_error("mqtt_connect", f"MQTT connect failed: {reason_code}")
            return
        self.metrics.set_mqtt_connected(True)
        self.metrics.set_ready(True)
        client.subscribe(self.settings.subscribe_topic, qos=self.settings.mqtt_qos)

    def _on_disconnect(self, _client: Any, _userdata: Any, *args: Any) -> None:
        reason_code = args[1] if len(args) >= 2 else args[0] if args else "unknown"
        self.metrics.set_mqtt_connected(False)
        self.metrics.set_ready(False)
        self.metrics.mark_error("mqtt_disconnect", f"MQTT disconnected: {reason_code}")

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        source_topic = str(message.topic)
        node_id = node_id_from_topic(source_topic) or "unknown"
        self.metrics.mark_flow_received(node_id=node_id, source_topic=source_topic)

        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("MQTT payload must be a JSON object")

            node_id = str(payload.get("node_id") or node_id_from_topic(source_topic) or "unknown")
            flow_id = str(payload.get("flow_id") or f"missing_flow_id_{int(time.time() * 1000)}")
            features = self.feature_adapter.vector_from_payload(payload)
            prediction = self._call_final_api(features)
            self._publish_prediction_and_alert(node_id, flow_id, source_topic, prediction)
        except json.JSONDecodeError as exc:
            self.metrics.mark_error("invalid_json", str(exc))
        except ValueError as exc:
            self.metrics.mark_error("invalid_message", str(exc))
        except httpx.HTTPError as exc:
            self.metrics.mark_error("final_api_http", str(exc))
        except Exception as exc:  # noqa: BLE001 - MQTT callbacks must not crash the process.
            self.metrics.mark_error("unexpected", str(exc))

    def _call_final_api(self, features: list[float]) -> dict[str, Any]:
        started = time.perf_counter()
        response = self.http.post("/predict", json={"features": features})
        self.metrics.observe_http_latency(time.perf_counter() - started)
        response.raise_for_status()
        return response.json()

    def _publish_prediction_and_alert(
        self,
        node_id: str,
        flow_id: str,
        source_topic: str,
        prediction: dict[str, Any],
    ) -> None:
        predicted_label_id = int(prediction.get("prediction", 0))
        predicted_label = "attack" if predicted_label_id == 1 else "normal"
        confidence = float(
            prediction.get("probability_attack")
            if predicted_label_id == 1
            else 1.0 - float(prediction.get("probability_attack", 0.0))
        )
        probability_attack = float(prediction.get("probability_attack", 0.0))
        is_alert = predicted_label_id == 1
        timestamp = utc_now()

        prediction_message = {
            "schema_version": "1.0",
            "event_type": "final_ids_prediction",
            "node_id": node_id,
            "timestamp": timestamp,
            "flow_id": flow_id,
            "predicted_label": predicted_label,
            "predicted_label_id": predicted_label_id,
            "confidence": confidence,
            "probability_attack": probability_attack,
            "is_alert": is_alert,
            "model_version": MODEL_VERSION,
            "selected_mask_id": SELECTED_MASK_ID,
            "features_count": int(prediction.get("selected_features_count", self.feature_adapter.selected_count)),
            "threshold": float(prediction.get("threshold", 0.4)),
            "source": SOURCE,
        }
        self._publish_json(f"ids/predictions/{node_id}", prediction_message)
        self.metrics.mark_prediction_published(node_id, predicted_label)

        if is_alert:
            severity = severity_for_attack(confidence)
            alert_message = {
                "schema_version": "1.0",
                "event_type": "final_ids_alert",
                "node_id": node_id,
                "timestamp": timestamp,
                "flow_id": flow_id,
                "predicted_label": "attack",
                "predicted_label_id": 1,
                "confidence": confidence,
                "severity": severity,
                "source_topic": source_topic,
                "model_version": MODEL_VERSION,
                "selected_mask_id": SELECTED_MASK_ID,
                "features_count": int(prediction.get("selected_features_count", self.feature_adapter.selected_count)),
                "source": SOURCE,
            }
            self._publish_json(f"ids/alerts/{node_id}", alert_message)
            self.metrics.mark_alert_published(node_id, severity)

    def _publish_json(self, topic: str, payload: dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("MQTT client is not initialized")
        result = self.client.publish(
            topic,
            json.dumps(payload, separators=(",", ":")),
            qos=self.settings.mqtt_qos,
        )
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"publish to {topic} failed with rc={result.rc}")

    def _ensure_mqtt(self) -> None:
        if self.mqtt is not None:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is required for final-mqtt-bridge") from exc
        self.mqtt = mqtt

    @staticmethod
    def _reason_code_value(reason_code: Any) -> int:
        value = getattr(reason_code, "value", reason_code)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0 if str(value).lower() == "success" else 1
