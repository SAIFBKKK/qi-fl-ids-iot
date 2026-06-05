"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
Phase 2 Live Lab - 2026-06-05.
Register the simulated UAV SITL node with the existing live-lab-controller.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from hardware_profiler import get_hardware_profile  # noqa: E402


NODE_ID = "iot-drone-sitl"
DEVICE_TYPE = "drone_sitl"
DECLARED_CPU_COUNT = 1
DECLARED_RAM_GB = 1.0
MQTT_TOPIC = f"ids/flows/{NODE_ID}"


def build_registration_payload(hostname: str | None = None) -> dict[str, object]:
    profile = get_hardware_profile()
    return {
        "node_id": NODE_ID,
        "hostname": hostname or str(profile.get("hostname") or socket.gethostname()),
        "cpu_count": DECLARED_CPU_COUNT,
        "ram_gb": DECLARED_RAM_GB,
        "device_type": DEVICE_TYPE,
        "mqtt_topic": MQTT_TOPIC,
    }


def post_registration(controller_url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{controller_url.rstrip('/')}/register-node",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register iot-drone-sitl with live-lab-controller.")
    parser.add_argument("--controller-url", default="http://192.168.56.1:8020")
    parser.add_argument("--dry-run", action="store_true", help="Print registration payload without HTTP request.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_registration_payload()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "payload": payload}, indent=2))
        return 0
    response = post_registration(args.controller_url, payload)
    print(json.dumps({"registered": True, "payload": payload, "controller_response": response}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
