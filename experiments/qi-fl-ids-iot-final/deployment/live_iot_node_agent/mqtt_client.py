from __future__ import annotations

import json
from typing import Any, Callable


AlertHandler = Callable[[str, dict[str, Any]], None]


class LiveLabMQTTClient:
    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str | None = "ids_user",
        password: str | None = None,
        client_id: str = "p16-live-iot-node-agent",
        qos: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.qos = qos
        self.mqtt = None
        self.client = None

    def connect(self, alert_topic: str, on_alert: AlertHandler) -> None:
        self._ensure_mqtt()
        try:
            self.client = self.mqtt.Client(self.mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        except (AttributeError, TypeError):
            self.client = self.mqtt.Client(client_id=self.client_id)
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = lambda client, _userdata, _flags, _reason, *_args: client.subscribe(alert_topic, qos=self.qos)
        self.client.on_message = lambda _client, _userdata, message: on_alert(message.topic, self._decode(message.payload))
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()

    def publish_flow(self, topic: str, payload: dict[str, object]) -> None:
        if self.client is None:
            raise RuntimeError("MQTT client is not connected")
        result = self.client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=self.qos)
        result.wait_for_publish()
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"publish to {topic} failed with rc={result.rc}")

    def stop(self) -> None:
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()

    @staticmethod
    def _decode(payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return {"raw": payload.decode("utf-8", errors="replace")}
        return value if isinstance(value, dict) else {"payload": value}

    def _ensure_mqtt(self) -> None:
        if self.mqtt is not None:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is required for live_iot_node_agent MQTT mode") from exc
        self.mqtt = mqtt

