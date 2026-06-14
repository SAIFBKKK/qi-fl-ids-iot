#!/usr/bin/env python
"""
test_publish.py - Publie des flows de test vers MQTT depuis un demo subset.

Usage:
    python services/scripts/test_publish.py --subset normal_traffic --count 5 --node-id node1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import paho.mqtt.client as mqtt

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = (
    REPO_ROOT
    / "experiments"
    / "fl-iot-ids-v3"
    / "outputs"
    / "deployment"
    / "baseline_fedavg_normal_classweights"
)
SUBSETS_DIR = REPO_ROOT / "data" / "cic-iot-2023" / "demo_subsets"
SERVICES_ENV = REPO_ROOT / "services" / ".env"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def config_value(env_file: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.getenv(key) or env_file.get(key) or default


def resolve_host_broker(broker: str) -> str:
    # services/.env uses the Docker DNS name. Host-side manual publishing needs localhost.
    return "localhost" if broker == "mosquitto" else broker


def build_client(env_file: dict[str, str]) -> mqtt.Client:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_publisher")
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id="test_publisher")

    username = config_value(env_file, "MQTT_USERNAME", "ids_user")
    password = config_value(env_file, "MQTT_PASSWORD")
    if not password:
        raise RuntimeError("MQTT_PASSWORD is required in services/.env or environment")

    client.username_pw_set(username, password)
    return client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="normal_traffic")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--node-id", default="node1")
    parser.add_argument("--rate", type=float, default=1.0, help="flows per second")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.rate <= 0:
        raise ValueError("--rate must be positive")

    env_file = load_env_file(SERVICES_ENV)
    feature_names = joblib.load(BUNDLE_PATH / "feature_names.pkl")

    subset_path = SUBSETS_DIR / f"{args.subset}.parquet"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {subset_path}")

    df = pd.read_parquet(subset_path)
    samples = df.sample(min(args.count, len(df)), random_state=42)

    client = build_client(env_file)
    broker = resolve_host_broker(config_value(env_file, "MQTT_BROKER", "localhost") or "localhost")
    port = int(config_value(env_file, "MQTT_PORT", "1883") or "1883")
    client.connect(broker, port)

    topic = f"ids/flows/{args.node_id}"
    delay = 1.0 / args.rate

    for idx, row in samples.iterrows():
        flow_msg = {
            "schema_version": "1.0",
            "flow_id": f"test_{idx}",
            "node_id": args.node_id,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "features": {name: float(row[name]) for name in feature_names},
            "ground_truth_label_id": int(row["label_id"]) if "label_id" in row else None,
        }
        client.publish(topic, json.dumps(flow_msg, separators=(",", ":")))
        print(f"Published flow {flow_msg['flow_id']} to {topic}")
        time.sleep(delay)

    client.disconnect()


if __name__ == "__main__":
    main()
