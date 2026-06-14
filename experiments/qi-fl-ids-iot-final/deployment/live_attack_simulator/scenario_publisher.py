from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from typing import Any, Iterable

from replay_loader import features_from_row, load_replay_rows
from scenario_profiles import ALLOWED_SCENARIOS, INPUT_MODE_LENGTHS, generate_controlled_features


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_payload(
    node_id: str,
    input_mode: str,
    scenario: str,
    sequence: int,
    features: list[float] | None = None,
) -> dict[str, object]:
    vector = features if features is not None else generate_controlled_features(scenario, input_mode, sequence)
    expected = INPUT_MODE_LENGTHS[input_mode]
    if len(vector) != expected:
        raise ValueError(f"{input_mode} expects {expected} features, got {len(vector)}")
    return {
        "flow_id": f"sim-{node_id}-{scenario}-{input_mode}-{sequence:06d}",
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "scenario": scenario,
        "features": [float(value) for value in vector],
    }


def replay_payloads(node_id: str, input_mode: str, scenario: str, count: int, replay_file: str | None) -> Iterable[dict[str, object]]:
    if replay_file:
        for sequence, row in enumerate(load_replay_rows(replay_file), start=1):
            row_scenario = str(row.get("scenario") or scenario)
            row_mode = str(row.get("input_mode") or input_mode)
            yield build_payload(node_id, row_mode, row_scenario, sequence, features_from_row(row))
            if sequence >= count:
                return
        return

    for sequence in range(1, count + 1):
        yield build_payload(node_id, input_mode, scenario, sequence)


def publish_payloads(args: argparse.Namespace) -> int:
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("paho-mqtt is required for scenario publishing") from exc

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id)
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=args.client_id)
    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
    client.loop_start()
    topic = args.topic or f"ids/flows/{args.node_id}"

    try:
        for payload in replay_payloads(args.node_id, args.input_mode, args.scenario, args.count, args.replay_file):
            result = client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=args.qos)
            result.wait_for_publish()
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"publish failed with rc={result.rc}")
            print(f"[scenario] published {payload['flow_id']} to {topic}", flush=True)
            time.sleep(max(args.interval_sec, 0.0))
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish controlled P16 feature-flow scenarios to MQTT.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--mqtt-host", required=True)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME", "ids_user"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--client-id", default="p16-live-attack-simulator")
    parser.add_argument("--qos", type=int, default=1)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODE_LENGTHS), default="original_28_scaled")
    parser.add_argument("--scenario", choices=ALLOWED_SCENARIOS, default="benign")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--topic")
    parser.add_argument("--replay-file")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(publish_payloads(parse_args()))

