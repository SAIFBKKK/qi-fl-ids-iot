from __future__ import annotations

import argparse
import json


def dry_run_demo_sequence() -> list[str]:
    return [
        "check server ports",
        "register weak node",
        "register medium node",
        "publish controlled 12-feature replay",
        "publish controlled 28-feature replay",
        "observe MQTT predictions and alerts",
        "collect dashboard and Grafana evidence",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run live lab demo sequence.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        print(json.dumps({"error": "Only --dry-run is enabled in P16.1 step 0."}, indent=2))
        return 2
    print(json.dumps({"mode": "dry_run", "steps": dry_run_demo_sequence()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

