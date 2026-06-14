from __future__ import annotations

import argparse
import json


def build_mqtt_connection_check(host: str, port: int) -> dict[str, object]:
    return {"mqtt_host": host, "mqtt_port": port, "mode": "dry_run", "status": "not_contacted"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run MQTT connection check.")
    parser.add_argument("--mqtt-host", default="SERVER_IP")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_mqtt_connection_check(args.mqtt_host, args.mqtt_port), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

