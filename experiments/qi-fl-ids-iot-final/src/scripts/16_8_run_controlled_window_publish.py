from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REALTIME_AGENT_ROOT = REPO_ROOT / "experiments" / "live_lab" / "realtime_agent"
if str(REALTIME_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_AGENT_ROOT))

from run_realtime_window_agent import run_agent  # noqa: E402


SUPPORTED_NODES = ("iot-rpi-weak", "iot-smart-watch-medium")
SUPPORTED_INPUT_MODES = ("selected_12_scaled", "original_28_scaled")


def build_agent_args(args: argparse.Namespace) -> argparse.Namespace:
    dry_run = args.dry_run or not args.publish
    return argparse.Namespace(
        node_id=args.node_id,
        broker=args.broker,
        port=args.port,
        username=args.username,
        password=args.password,
        input_mode=args.input_mode,
        window_size=args.window_size,
        dry_run=dry_run,
        publish=args.publish,
        pcap=None,
        interface=None,
        allow_live_capture=False,
        max_windows=args.max_windows,
        sleep_ms=args.sleep_ms,
        no_scale=False,
        summary_output=None,
    )


def summarize_agent_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    window = agent_result.get("windows", [{}])[0] if agent_result.get("windows") else {}
    publish_result = window.get("publish_result", {})
    payload = publish_result.get("payload", {})
    summary = {
        "ok": bool(agent_result.get("ok")) and (agent_result.get("dry_run") or bool(publish_result.get("published"))),
        "phase": "P16.8",
        "published": bool(publish_result.get("published")),
        "dry_run": bool(agent_result.get("dry_run")),
        "node_id": agent_result.get("node_id"),
        "topic": publish_result.get("topic"),
        "input_mode": agent_result.get("input_mode"),
        "features_count": int(window.get("feature_count", 0) or 0),
        "window_size": agent_result.get("window_size"),
        "flow_id": payload.get("flow_id") or window.get("window_summary", {}).get("flow_id"),
        "payload": payload,
        "window_summary": window.get("window_summary", {}),
        "transform": window.get("transform", {}),
        "warnings": agent_result.get("warnings", []),
    }
    if publish_result.get("error"):
        summary["error"] = publish_result["error"]
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P16.8 controlled SyntheticPacketSource window publication to MQTT."
    )
    parser.add_argument("--broker", default="192.168.56.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--username", default="ids_user")
    parser.add_argument("--password", default="changeme_in_dotenv")
    parser.add_argument("--node-id", choices=SUPPORTED_NODES, required=True)
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agent_result = run_agent(build_agent_args(args))
    summary = summarize_agent_result(agent_result)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
