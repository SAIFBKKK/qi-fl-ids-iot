from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "nodes" / "common"
sys.path.insert(0, str(COMMON_DIR))

from hardware_profiler import get_hardware_profile  # noqa: E402


def build_registration_payload(node_id: str, device_type: str) -> dict[str, object]:
    profile = get_hardware_profile()
    return {
        "node_id": node_id,
        "hostname": profile["hostname"],
        "cpu_count": profile["cpu_count"],
        "ram_gb": profile["ram_gb"],
        "device_type": device_type,
        "mqtt_topic": f"ids/flows/{node_id}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run node registration payload builder.")
    parser.add_argument("--node-id", default="iot-rpi-weak")
    parser.add_argument("--device-type", default="raspberry_like")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_registration_payload(args.node_id, args.device_type), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

