from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from metrics import OnlineValidatorMetrics


@dataclass(frozen=True)
class MQTTSettings:
    broker: str
    port: int
    username: str | None
    password: str | None
    topics: tuple[str, ...]
    client_id: str

    @classmethod
    def from_env(cls) -> "MQTTSettings":
        topics_raw = os.getenv(
            "OBSERVE_TOPICS",
            "ids/flows/#,ids/predictions/#,ids/alerts/#,ids/status/#",
        )
        topics = tuple(topic.strip() for topic in topics_raw.split(",") if topic.strip())
        return cls(
            broker=os.getenv("MQTT_BROKER", "mosquitto"),
            port=int(os.getenv("MQTT_PORT", "1883")),
            username=os.getenv("MQTT_USERNAME", "ids_user"),
            password=os.getenv("MQTT_PASSWORD"),
            topics=topics,
            client_id=os.getenv("MQTT_CLIENT_ID", "p15-online-validator"),
        )


class MQTTObserver:
    def __init__(self, settings: MQTTSettings, metrics: OnlineValidatorMetrics) -> None:
        self.settings = settings
        self.metrics = metrics
        self.mqtt = None
        self.client = None

    def start(self) -> None:
        self._ensure_mqtt()
        try:
            self.client = self.mqtt.Client(self.mqtt.CallbackAPIVersion.VERSION2, client_id=self.settings.client_id)
        except (AttributeError, TypeError):
            self.client = self.mqtt.Client(client_id=self.settings.client_id)

        if self.settings.username:
            self.client.username_pw_set(self.settings.username, self.settings.password)

        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.connect_async(self.settings.broker, self.settings.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.metrics.set_connected(False)

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        if self._reason_code_value(reason_code) != 0:
            self.metrics.set_connected(False)
            self.metrics.mark_error(f"mqtt connect failed: {reason_code}")
            return
        self.metrics.set_connected(True)
        for topic in self.settings.topics:
            client.subscribe(topic, qos=1)

    def _on_disconnect(self, _client: Any, _userdata: Any, *args: Any) -> None:
        self.metrics.set_connected(False)
        reason_code = args[1] if len(args) >= 2 else args[0] if args else "unknown"
        self.metrics.mark_error(f"mqtt disconnected: {reason_code}")

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        self.metrics.record_message(message.topic, message.payload)

    @staticmethod
    def _reason_code_value(reason_code: Any) -> int:
        value = getattr(reason_code, "value", reason_code)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0 if str(value).lower() == "success" else 1

    def _ensure_mqtt(self) -> None:
        if self.mqtt is not None:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is required for online-validator") from exc
        self.mqtt = mqtt
