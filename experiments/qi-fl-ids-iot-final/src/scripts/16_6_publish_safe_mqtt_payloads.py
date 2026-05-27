from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


SELECTED_MASK_ID = "conservative_seed_42"
SUPPORTED_NODES = ("iot-rpi-weak", "iot-smart-watch-medium")
SUPPORTED_INPUT_MODES = ("selected_12_scaled", "original_28_scaled")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def feature_count_for_mode(input_mode: str) -> int:
    if input_mode == "selected_12_scaled":
        return 12
    if input_mode == "original_28_scaled":
        return 28
    raise ValueError(f"unsupported input mode: {input_mode}")


def build_topic(node_id: str) -> str:
    return f"ids/flows/{node_id}"


def build_payload(node_id: str, input_mode: str, flow_id: str | None = None) -> dict[str, Any]:
    feature_count = feature_count_for_mode(input_mode)
    return {
        "flow_id": flow_id or f"p16-6-safe-{uuid4().hex[:12]}",
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "features": [0.0 for _ in range(feature_count)],
        "metadata": {
            "phase": "P16.6",
            "safe_payload": True,
            "selected_mask_id": SELECTED_MASK_ID,
            "purpose": "live MQTT node-to-IDS pipeline validation",
            "traffic_source": "controlled_json_payload_only",
        },
    }


def publish_with_paho(
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

    client = mqtt.Client(client_id=f"p16-6-safe-publisher-{uuid4().hex[:8]}")
    if username:
        client.username_pw_set(username, password)
    client.connect(broker, port, keepalive=30)
    result = client.publish(topic, json.dumps(payload), qos=0, retain=False)
    result.wait_for_publish(timeout=10)
    client.disconnect()
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"MQTT publish failed with rc={result.rc}")


def publish_with_mosquitto_pub(
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    topic: str,
    payload: dict[str, Any],
) -> None:
    mosquitto_pub = shutil.which("mosquitto_pub")
    if not mosquitto_pub:
        raise RuntimeError("neither paho-mqtt nor mosquitto_pub is available")

    command = [
        mosquitto_pub,
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


def publish_payload(args: argparse.Namespace, topic: str, payload: dict[str, Any]) -> str:
    try:
        publish_with_paho(args.broker, args.port, args.username, args.password, topic, payload)
        return "paho-mqtt"
    except Exception as paho_error:
        try:
            publish_with_mosquitto_pub(args.broker, args.port, args.username, args.password, topic, payload)
            return "mosquitto_pub"
        except Exception as fallback_error:
            raise RuntimeError(
                f"MQTT publish failed with paho-mqtt ({paho_error}) and mosquitto_pub ({fallback_error})"
            ) from fallback_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a controlled P16.6 JSON flow payload to the live lab MQTT broker."
    )
    parser.add_argument("--broker", default="192.168.56.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--username", default="ids_user")
    parser.add_argument("--password", default="changeme_in_dotenv")
    parser.add_argument("--node-id", choices=SUPPORTED_NODES, required=True)
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, required=True)
    parser.add_argument("--flow-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args.node_id, args.input_mode, args.flow_id)
    topic = build_topic(args.node_id)
    output: dict[str, Any] = {
        "dry_run": args.dry_run,
        "broker": args.broker,
        "port": args.port,
        "topic": topic,
        "payload": payload,
    }

    if args.dry_run:
        print(json.dumps(output, indent=2))
        return 0

    try:
        publisher = publish_payload(args, topic, payload)
    except Exception as exc:
        print(json.dumps({**output, "published": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({**output, "published": True, "publisher": publisher}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
