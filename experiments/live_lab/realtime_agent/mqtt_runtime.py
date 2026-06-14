from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_flow_payload(
    *,
    node_id: str,
    input_mode: str,
    features: list[float],
    flow_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "flow_id": flow_id or f"p16-7-window-{uuid4().hex[:12]}",
        "node_id": node_id,
        "timestamp": timestamp or utc_now(),
        "input_mode": input_mode,
        "scenario": "p16_7_realtime_window",
        "features": [float(value) for value in features],
    }


def publish_with_paho(
    *,
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    topic: str,
    payload: dict[str, Any],
) -> None:
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("paho-mqtt is not installed") from exc

    client = mqtt.Client(client_id=f"p16-7-realtime-agent-{uuid4().hex[:8]}")
    if username:
        client.username_pw_set(username, password)
    client.connect(broker, port, keepalive=30)
    result = client.publish(topic, json.dumps(payload), qos=0, retain=False)
    result.wait_for_publish(timeout=10)
    client.disconnect()
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"MQTT publish failed with rc={result.rc}")


def publish_with_mosquitto_pub(
    *,
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    topic: str,
    payload: dict[str, Any],
) -> None:
    executable = shutil.which("mosquitto_pub")
    if not executable:
        raise RuntimeError("mosquitto_pub is not available")
    command = [
        executable,
        "-h",
        broker,
        "-p",
        str(port),
        "-t",
        topic,
        "-m",
        json.dumps(payload),
    ]
    if username:
        command.extend(["-u", username])
    if password:
        command.extend(["-P", password])
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "mosquitto_pub failed")


def publish_flow_payload(
    *,
    node_id: str,
    input_mode: str,
    features: list[float],
    broker: str = "192.168.56.1",
    port: int = 1883,
    username: str | None = "ids_user",
    password: str | None = "changeme_in_dotenv",
    dry_run: bool = True,
    publish: bool = False,
    flow_id: str | None = None,
) -> dict[str, Any]:
    topic = f"ids/flows/{node_id}"
    payload = build_flow_payload(node_id=node_id, input_mode=input_mode, features=features, flow_id=flow_id)
    result: dict[str, Any] = {
        "topic": topic,
        "payload": payload,
        "dry_run": dry_run,
        "publish_requested": publish,
        "published": False,
    }
    if dry_run or not publish:
        return result
    try:
        publish_with_paho(
            broker=broker,
            port=port,
            username=username,
            password=password,
            topic=topic,
            payload=payload,
        )
        result.update({"published": True, "publisher": "paho-mqtt"})
        return result
    except Exception as paho_error:
        try:
            publish_with_mosquitto_pub(
                broker=broker,
                port=port,
                username=username,
                password=password,
                topic=topic,
                payload=payload,
            )
            result.update({"published": True, "publisher": "mosquitto_pub"})
            return result
        except Exception as fallback_error:
            result.update({"published": False, "error": f"{paho_error}; {fallback_error}"})
            return result
