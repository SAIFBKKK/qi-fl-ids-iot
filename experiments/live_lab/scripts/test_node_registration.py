from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


COMMON_DIR = Path(__file__).resolve().parents[1] / "nodes" / "common"
sys.path.insert(0, str(COMMON_DIR))

from hardware_profiler import get_hardware_profile  # noqa: E402


def default_declared_profile(node_id: str) -> dict[str, object]:
    if node_id == "iot-drone-sitl":
        return {
            "cpu_count": 1,
            "ram_gb": 1.0,
            "device_type": "drone_sitl",
            "mqtt_topic": "ids/flows/iot-drone-sitl",
        }
    if node_id == "iot-smart-watch-medium":
        return {
            "cpu_count": 1,
            "ram_gb": 1.5,
            "device_type": "smart_watch_like",
            "mqtt_topic": "ids/flows/iot-smart-watch-medium",
        }
    return {
        "cpu_count": 1,
        "ram_gb": 1.0,
        "device_type": "unknown",
        "mqtt_topic": f"ids/flows/{node_id}",
    }


def build_registration_payload(
    node_id: str,
    hostname: str | None = None,
    cpu_count: int | None = None,
    ram_gb: float | None = None,
    device_type: str | None = None,
    mqtt_topic: str | None = None,
) -> dict[str, object]:
    profile = get_hardware_profile()
    declared = default_declared_profile(node_id)
    return {
        "node_id": node_id,
        "hostname": hostname or str(profile["hostname"]),
        "cpu_count": int(cpu_count if cpu_count is not None else declared["cpu_count"]),
        "ram_gb": float(ram_gb if ram_gb is not None else declared["ram_gb"]),
        "device_type": str(device_type or declared["device_type"]),
        "mqtt_topic": str(mqtt_topic or declared["mqtt_topic"]),
    }


def post_registration(server_url: str, payload: dict[str, object]) -> dict[str, Any]:
    url = f"{server_url.rstrip('/')}/register-node"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"registration failed with HTTP {exc.code}: {body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or send a live-lab-controller node registration payload.")
    parser.add_argument("--server-url", default="http://192.168.56.1:8020")
    parser.add_argument("--node-id", default="iot-drone-sitl")
    parser.add_argument("--hostname")
    parser.add_argument("--cpu-count", type=int)
    parser.add_argument("--ram-gb", type=float)
    parser.add_argument("--device-type")
    parser.add_argument("--mqtt-topic")
    parser.add_argument("--register", action="store_true", help="POST the payload to {server_url}/register-node.")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload without network activity.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_registration_payload(
        node_id=args.node_id,
        hostname=args.hostname,
        cpu_count=args.cpu_count,
        ram_gb=args.ram_gb,
        device_type=args.device_type,
        mqtt_topic=args.mqtt_topic,
    )
    if args.register:
        print(json.dumps(post_registration(args.server_url, payload), indent=2))
        return 0
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

