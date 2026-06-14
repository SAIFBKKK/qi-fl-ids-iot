from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "nodes" / "common"
sys.path.insert(0, str(COMMON_DIR))

from mqtt_client import build_payload, build_topic  # noqa: E402
from scenario_profiles import scenario_names  # noqa: E402


def build_dry_run_scenario(node_id: str, input_mode: str, scenario: str) -> dict[str, object]:
    feature_count = 12 if input_mode == "selected_12_scaled" else 28
    return {
        "topic": build_topic(node_id, "flows"),
        "payload": build_payload(node_id, input_mode, [0.0 for _ in range(feature_count)], scenario=scenario),
        "allowed_scenarios": scenario_names(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run scenario publisher placeholder.")
    parser.add_argument("--node-id", default="iot-drone-sitl")
    parser.add_argument("--input-mode", choices=["selected_12_scaled", "original_28_scaled"], default="selected_12_scaled")
    parser.add_argument("--scenario", choices=scenario_names(), default="benign")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        print("Only --dry-run is enabled in P16.1 step 0.", file=sys.stderr)
        return 2
    print(json.dumps(build_dry_run_scenario(args.node_id, args.input_mode, args.scenario), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


