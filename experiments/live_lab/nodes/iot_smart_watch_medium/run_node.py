from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
REALTIME_DIR = Path(__file__).resolve().parents[2] / "realtime_agent"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(REALTIME_DIR))

from edge_inference import EdgeInferenceRuntime  # noqa: E402
from feature_schema import expected_28_features  # noqa: E402
from hardware_profiler import get_hardware_profile  # noqa: E402
from mqtt_client import build_payload, build_topic  # noqa: E402
from qga_mask import selected_mask_id  # noqa: E402
from utils import load_simple_yaml  # noqa: E402


def build_dry_run_summary(config_path: str | Path | None = None) -> dict[str, object]:
    config = load_simple_yaml(config_path or Path(__file__).with_name("config.yaml"))
    node_id = str(config.get("node_id", "iot-smart-watch-medium"))
    features = [0.0 for _ in expected_28_features()]
    edge_runtime = EdgeInferenceRuntime(model_path=None, enabled=False)
    return {
        "node_id": node_id,
        "device_type": config.get("device_type", "smart_watch_like"),
        "inference_mode": config.get("inference_mode", "edge_placeholder"),
        "input_mode": config.get("input_mode", "original_28_scaled"),
        "selected_mask_id": selected_mask_id(),
        "publish_topic": build_topic(node_id, "flows"),
        "alert_topic": build_topic(node_id, "alerts"),
        "hardware_profile": get_hardware_profile(),
        "edge_runtime": edge_runtime.describe(),
        "sample_payload": build_payload(node_id, "original_28_scaled", features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run smart-watch-like IoT node placeholder.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--dry-run", action="store_true", help="Print planned node configuration without network activity.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        print("Only --dry-run is enabled in P16.1 step 0.", file=sys.stderr)
        return 2
    print(json.dumps(build_dry_run_summary(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

