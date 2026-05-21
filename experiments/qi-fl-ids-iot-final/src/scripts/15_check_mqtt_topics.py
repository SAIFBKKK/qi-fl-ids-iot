"""Observe IDS MQTT topics for P15 evidence collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_DIR = Path("experiments/qi-fl-ids-iot-final")
DEFAULT_REPORTS_DIR = FINAL_DIR / "outputs" / "reports"


def read_services_env_password() -> str | None:
    if os.getenv("MQTT_PASSWORD"):
        return os.getenv("MQTT_PASSWORD")
    env_path = Path("services/.env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("MQTT_PASSWORD="):
            return line.split("=", 1)[1].strip()
    return None


def topic_family(topic: str) -> str:
    if topic.startswith("ids/flows/"):
        return "flows"
    if topic.startswith("ids/predictions/"):
        return "predictions"
    if topic.startswith("ids/alerts/"):
        return "alerts"
    if topic.startswith("ids/status/"):
        return "status"
    return "other"


def write_csv(path: Path, topic_counts: Counter[str], family_counts: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "name", "count"])
        writer.writeheader()
        for family, count in sorted(family_counts.items()):
            writer.writerow({"kind": "family", "name": family, "count": count})
        for topic, count in sorted(topic_counts.items()):
            writer.writerow({"kind": "topic", "name": topic, "count": count})


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe ids/# MQTT topics for a bounded duration.")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--username", default=os.getenv("MQTT_USERNAME", "ids_user"))
    parser.add_argument("--password", default=read_services_env_password())
    parser.add_argument("--topic", default="ids/#")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    topic_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    connected = threading.Event()
    errors: list[str] = []

    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        summary = {
            "accepted": False,
            "error": f"paho-mqtt unavailable: {exc}",
            "topic_counts": {},
            "family_counts": {},
        }
        (args.reports_dir / "p15_mqtt_topics_observed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(args.reports_dir / "p15_mqtt_topics_observed.csv", topic_counts, family_counts)
        print(json.dumps(summary, indent=2))
        return 2

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"p15-topic-check-{int(time.time())}")
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=f"p15-topic-check-{int(time.time())}")

    if args.username:
        client.username_pw_set(args.username, args.password)

    def on_connect(client_obj: Any, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        value = getattr(reason_code, "value", reason_code)
        if str(value) not in {"0", "Success"}:
            errors.append(f"connect failed: {reason_code}")
            return
        connected.set()
        client_obj.subscribe(args.topic, qos=1)

    def on_message(_client: Any, _userdata: Any, message: Any) -> None:
        payload = message.payload.decode("utf-8", errors="replace")
        topic_counts[message.topic] += 1
        family_counts[topic_family(message.topic)] += 1
        if len(samples) < 20:
            samples.append(
                {
                    "topic": message.topic,
                    "payload_preview": payload[:250],
                    "received_at_unix": time.time(),
                }
            )

    client.on_connect = on_connect
    client.on_message = on_message

    started_at = time.time()
    try:
        client.connect(args.broker, args.port, keepalive=30)
        client.loop_start()
        connected.wait(timeout=5)
        time.sleep(max(args.duration, 0.0))
    except Exception as exc:  # noqa: BLE001 - evidence script must write a report.
        errors.append(str(exc))
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    summary = {
        "phase": "P15",
        "mode": "mqtt_topic_observation",
        "broker": args.broker,
        "port": args.port,
        "topic": args.topic,
        "duration_sec": round(time.time() - started_at, 3),
        "connected": connected.is_set(),
        "topic_counts": dict(topic_counts),
        "family_counts": dict(family_counts),
        "samples": samples,
        "errors": errors,
        "accepted": connected.is_set() and not errors,
    }
    (args.reports_dir / "p15_mqtt_topics_observed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.reports_dir / "p15_mqtt_topics_observed.csv", topic_counts, family_counts)
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
