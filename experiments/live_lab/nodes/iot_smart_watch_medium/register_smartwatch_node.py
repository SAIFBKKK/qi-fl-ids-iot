"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
P16.17 - Continuous Smartwatch Traffic Observer.
Safe registration helper for the iot-smart-watch-medium live-lab node.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from hardware_profiler import get_hardware_profile  # noqa: E402


NODE_ID = "iot-smart-watch-medium"
DEVICE_TYPE = "smart_watch"
DEFAULT_CONTROLLER_URL = "http://192.168.56.1:8020"


def build_registration_payload() -> dict[str, Any]:
    profile = get_hardware_profile()
    return {
        "node_id": NODE_ID,
        "hostname": str(profile.get("hostname", NODE_ID)),
        "cpu_count": int(profile.get("cpu_count") or 1),
        "ram_gb": float(profile.get("ram_gb") or 0.0),
        "device_type": DEVICE_TYPE,
        "mqtt_topic": f"ids/flows/{NODE_ID}",
    }


def post_registration(controller_url: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Register iot-smart-watch-medium with live-lab-controller.")
    parser.add_argument("--controller-url", default=DEFAULT_CONTROLLER_URL)
    parser.add_argument("--dry-run", action="store_true", help="Print the registration payload without network activity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_registration_payload()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "controller_url": args.controller_url, "payload": payload}, indent=2))
        return 0
    response = post_registration(args.controller_url, payload)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
