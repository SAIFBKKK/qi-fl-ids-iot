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
from mqtt_client import build_payload, build_topic  # noqa: E402
from qga_mask import selected_features, selected_mask_id  # noqa: E402
from utils import load_simple_yaml  # noqa: E402


NODE_ID = "iot-rpi-weak"
DECLARED_CPU_COUNT = 1
DECLARED_RAM_GB = 1.0
DEVICE_TYPE = "raspberry_like"


def build_registration_payload(hostname: str | None = None) -> dict[str, object]:
    profile = get_hardware_profile()
    return {
        "node_id": NODE_ID,
        "hostname": hostname or str(profile["hostname"]),
        "cpu_count": DECLARED_CPU_COUNT,
        "ram_gb": DECLARED_RAM_GB,
        "device_type": DEVICE_TYPE,
        "mqtt_topic": f"ids/flows/{NODE_ID}",
    }


def post_registration(server_url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/register-node",
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


def build_dry_run_summary(config_path: str | Path | None = None) -> dict[str, object]:
    config = load_simple_yaml(config_path or Path(__file__).with_name("config.yaml"))
    node_id = str(config.get("node_id", NODE_ID))
    features = [0.0 for _ in selected_features()]
    return {
        "node_id": node_id,
        "device_type": config.get("device_type", DEVICE_TYPE),
        "inference_mode": config.get("inference_mode", "server_side"),
        "input_mode": config.get("input_mode", "selected_12_scaled"),
        "selected_mask_id": selected_mask_id(),
        "publish_topic": build_topic(node_id, "flows"),
        "alert_topic": build_topic(node_id, "alerts"),
        "hardware_profile": get_hardware_profile(),
        "registration_payload": build_registration_payload(),
        "sample_payload": build_payload(node_id, "selected_12_scaled", features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run weak IoT node placeholder.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--server-url", default="http://192.168.56.1:8020")
    parser.add_argument("--register", action="store_true", help="Register this node with live-lab-controller.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned node configuration without network activity.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.register:
        response = post_registration(args.server_url, build_registration_payload())
        print(json.dumps(response, indent=2))
        return 0
    if not args.dry_run:
        print("Use --dry-run to inspect payloads or --register to call /register-node.", file=sys.stderr)
        return 2
    print(json.dumps(build_dry_run_summary(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
