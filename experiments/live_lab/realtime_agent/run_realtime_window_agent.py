from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from feature_extractor import extract_28_features_from_window
from flow_window import PacketWindow, packet_window_from_packets
from mqtt_runtime import publish_flow_payload
from packet_capture import choose_packet_source
from scaler_runtime import apply_qga_mask, load_optional_scaler, load_qga_mask, scale_28_features


SUPPORTED_INPUT_MODES = ("selected_12_scaled", "original_28_scaled", "original_28_unscaled")


def select_features_for_mode(
    *,
    input_mode: str,
    features_28: list[float],
    no_scale: bool,
) -> tuple[list[float], dict[str, Any]]:
    scaler = load_optional_scaler()
    mask = load_qga_mask()
    warnings: list[str] = []
    if input_mode == "original_28_unscaled":
        return [float(value) for value in features_28], {
            "scaler": {"available": scaler.available, "used": False, "warnings": scaler.warnings},
            "qga_mask": {"used": False, "mask_id": mask["mask_id"], "warnings": mask["warnings"]},
            "warnings": warnings,
        }

    scaled = scale_28_features(features_28, scaler=scaler, no_scale=no_scale)
    warnings.extend(scaler.warnings)
    if input_mode == "original_28_scaled":
        return scaled, {
            "scaler": {**scaler.describe(), "used": not no_scale and scaler.available},
            "qga_mask": {"used": False, "mask_id": mask["mask_id"], "warnings": mask["warnings"]},
            "warnings": list(dict.fromkeys(warnings + mask["warnings"])),
        }

    selected = apply_qga_mask(scaled)
    warnings.extend(mask["warnings"])
    return selected, {
        "scaler": {**scaler.describe(), "used": not no_scale and scaler.available},
        "qga_mask": {
            "used": True,
            "mask_id": mask["mask_id"],
            "selected_indices": mask["selected_indices"],
            "warnings": mask["warnings"],
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def run_agent(args: argparse.Namespace) -> dict[str, Any]:
    effective_dry_run = args.dry_run or not args.publish
    no_scale = args.no_scale
    source = choose_packet_source(
        pcap=args.pcap,
        interface=args.interface,
        allow_live_capture=args.allow_live_capture,
        synthetic_packet_count=args.window_size * args.max_windows,
    )
    current = PacketWindow(window_size=args.window_size)
    windows: list[dict[str, Any]] = []
    warnings: list[str] = []
    max_packets = args.window_size * args.max_windows

    for packet in source.iter_packets(max_packets=max_packets):
        if not current.add_packet(packet):
            continue
        flow_id = current.flow_id
        completed_packets = current.flush()
        completed = packet_window_from_packets(
            completed_packets,
            window_size=args.window_size,
            flow_id=flow_id,
        )
        current = PacketWindow(window_size=args.window_size)
        extraction = extract_28_features_from_window(completed)
        output_features, transform_info = select_features_for_mode(
            input_mode=args.input_mode,
            features_28=extraction["features_28"],
            no_scale=no_scale,
        )
        warnings.extend(transform_info.get("warnings", []))
        publish_result = publish_flow_payload(
            node_id=args.node_id,
            input_mode=args.input_mode,
            features=output_features,
            broker=args.broker,
            port=args.port,
            username=args.username,
            password=args.password,
            dry_run=effective_dry_run,
            publish=args.publish,
            flow_id=completed.flow_id,
        )
        windows.append(
            {
                "window_index": len(windows) + 1,
                "window_summary": completed.summary(),
                "feature_names": extraction["feature_names"],
                "feature_count": len(output_features),
                "features": output_features,
                "approximated_features": extraction["approximated_features"],
                "unsupported_features": extraction["unsupported_features"],
                "transform": transform_info,
                "publish_result": publish_result,
            }
        )
        if len(windows) >= args.max_windows:
            break
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000)

    if not windows:
        warnings.append("No complete packet window was produced.")

    result = {
        "ok": bool(windows),
        "phase": "P16.7",
        "node_id": args.node_id,
        "input_mode": args.input_mode,
        "dry_run": effective_dry_run,
        "publish": args.publish,
        "source": source.describe(),
        "window_size": args.window_size,
        "windows": windows,
        "warnings": list(dict.fromkeys(warnings)),
    }
    if args.summary_output:
        target = Path(args.summary_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["summary_output"] = str(target)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P16.7 safe packet-window feature extraction prototype.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--broker", default="192.168.56.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--username", default="ids_user")
    parser.add_argument("--password", default="changeme_in_dotenv")
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, default="selected_12_scaled")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--pcap", default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--allow-live-capture", action="store_true")
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args(argv)
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    if args.interface and not args.allow_live_capture and not args.pcap:
        parser.error("--interface requires --allow-live-capture")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_agent(args)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
