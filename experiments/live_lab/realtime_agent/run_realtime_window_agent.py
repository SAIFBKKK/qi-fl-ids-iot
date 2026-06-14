from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from feature_extractor import extract_28_features_from_window
from flow_window import PacketWindow, packet_window_from_packets
from mqtt_runtime import publish_flow_payload, publish_with_mosquitto_pub, publish_with_paho, utc_now
from packet_capture import OptionalPcapPacketSource, SyntheticPacketSource
from passive_capture import passive_packet_source
from scaler_runtime import apply_qga_mask, load_optional_scaler, load_qga_mask, scale_28_features


SUPPORTED_INPUT_MODES = ("selected_12_scaled", "original_28_scaled", "original_28_unscaled")
SUPPORTED_SOURCES = ("synthetic", "pcap", "live")


def ensure_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "source": "synthetic",
        "node_ip": None,
        "peer_ip": None,
        "timeout_seconds": 30,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


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
            "scaler": {**scaler.describe(), "used": False},
            "qga_mask": {
                "used": False,
                "mask_id": mask["mask_id"],
                "selected_mask_id": mask["selected_mask_id"],
                "warnings": mask["warnings"],
            },
            "warnings": warnings,
        }

    scaled = scale_28_features(features_28, scaler=scaler, no_scale=no_scale)
    warnings.extend(scaler.warnings)
    if input_mode == "original_28_scaled":
        return scaled, {
            "scaler": {**scaler.describe(), "used": not no_scale and scaler.available},
            "qga_mask": {
                "used": False,
                "mask_id": mask["mask_id"],
                "selected_mask_id": mask["selected_mask_id"],
                "warnings": mask["warnings"],
            },
            "warnings": list(dict.fromkeys(warnings + mask["warnings"])),
        }

    selected = apply_qga_mask(scaled)
    warnings.extend(mask["warnings"])
    return selected, {
        "scaler": {**scaler.describe(), "used": not no_scale and scaler.available},
        "qga_mask": {
            "used": True,
            "mask_id": mask["mask_id"],
            "selected_mask_id": mask["selected_mask_id"],
            "selected_indices": mask["selected_indices"],
            "warnings": mask["warnings"],
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def choose_source(args: argparse.Namespace) -> Any:
    if args.source == "pcap":
        if not args.pcap:
            raise ValueError("source=pcap requires --pcap")
        return OptionalPcapPacketSource(args.pcap)
    if args.source == "live":
        if not args.allow_live_capture:
            raise PermissionError("source=live requires --allow-live-capture")
        if not args.interface:
            raise ValueError("source=live requires --interface")
        if not args.node_ip:
            raise ValueError("source=live requires --node-ip")
        return passive_packet_source(
            interface_name=args.interface,
            max_packets=args.window_size * args.max_windows,
            timeout_seconds=args.timeout_seconds,
            node_ip=args.node_ip,
            peer_ip=args.peer_ip,
        )
    return SyntheticPacketSource(packet_count=args.window_size * args.max_windows)


def publish_real_traffic_payload(
    *,
    node_id: str,
    input_mode: str,
    features: list[float],
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    dry_run: bool,
    publish: bool,
    flow_id: str,
    capture_interface: str,
    node_ip: str,
    peer_ip: str | None,
    window_size: int,
) -> dict[str, Any]:
    topic = f"ids/flows/{node_id}"
    payload = {
        "flow_id": flow_id,
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "observation_mode": "real_traffic_passive_capture",
        "capture_interface": capture_interface,
        "node_ip": node_ip,
        "peer_ip": peer_ip,
        "window_size": window_size,
        "scenario": "real_traffic_observation",
        "features": [float(value) for value in features],
    }
    result: dict[str, Any] = {
        "topic": topic,
        "payload": payload,
        "dry_run": dry_run,
        "publish_requested": publish,
        "published": False,
    }
    if dry_run or not publish:
        return result
    try:
        publish_with_paho(
            broker=broker,
            port=port,
            username=username,
            password=password,
            topic=topic,
            payload=payload,
        )
        result.update({"published": True, "publisher": "paho-mqtt"})
        return result
    except Exception as paho_error:
        try:
            publish_with_mosquitto_pub(
                broker=broker,
                port=port,
                username=username,
                password=password,
                topic=topic,
                payload=payload,
            )
            result.update({"published": True, "publisher": "mosquitto_pub"})
            return result
        except Exception as fallback_error:
            result.update({"published": False, "error": f"{paho_error}; {fallback_error}"})
            return result


def publish_window_payload(
    *,
    args: argparse.Namespace,
    output_features: list[float],
    flow_id: str,
    effective_dry_run: bool,
) -> dict[str, Any]:
    if args.source == "live":
        return publish_real_traffic_payload(
            node_id=args.node_id,
            input_mode=args.input_mode,
            features=output_features,
            broker=args.broker,
            port=args.port,
            username=args.username,
            password=args.password,
            dry_run=effective_dry_run,
            publish=args.publish,
            flow_id=flow_id,
            capture_interface=args.interface,
            node_ip=args.node_ip,
            peer_ip=args.peer_ip,
            window_size=args.window_size,
        )
    return publish_flow_payload(
        node_id=args.node_id,
        input_mode=args.input_mode,
        features=output_features,
        broker=args.broker,
        port=args.port,
        username=args.username,
        password=args.password,
        dry_run=effective_dry_run,
        publish=args.publish,
        flow_id=flow_id,
    )


def run_agent(args: argparse.Namespace) -> dict[str, Any]:
    args = ensure_runtime_defaults(args)
    effective_dry_run = args.dry_run or not args.publish
    no_scale = args.no_scale
    source = choose_source(args)
    current = PacketWindow(window_size=args.window_size)
    windows: list[dict[str, Any]] = []
    warnings: list[str] = []
    max_packets = args.window_size * args.max_windows
    if args.source == "live":
        warnings.append("Passive observation only. No traffic is generated by this agent.")

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
        publish_result = publish_window_payload(
            args=args,
            output_features=output_features,
            flow_id=completed.flow_id,
            effective_dry_run=effective_dry_run,
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
        "phase": "P16.12" if args.source == "live" else "P16.7",
        "node_id": args.node_id,
        "input_mode": args.input_mode,
        "dry_run": effective_dry_run,
        "publish": args.publish,
        "observation_mode": "real_traffic_passive_capture" if args.source == "live" else None,
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
    parser = argparse.ArgumentParser(description="Safe packet-window feature extraction and passive observation agent.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--broker", default="192.168.56.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--username", default="ids_user")
    parser.add_argument("--password", default="changeme_in_dotenv")
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, default="selected_12_scaled")
    parser.add_argument("--source", choices=SUPPORTED_SOURCES, default="synthetic")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--pcap", default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--node-ip", default=None)
    parser.add_argument("--peer-ip", default=None)
    parser.add_argument("--allow-live-capture", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args(argv)
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.source == "pcap" and not args.pcap:
        parser.error("--source pcap requires --pcap")
    if args.source == "live":
        if not args.allow_live_capture:
            parser.error("--source live requires --allow-live-capture")
        if not args.interface:
            parser.error("--source live requires --interface")
        if not args.node_ip:
            parser.error("--source live requires --node-ip")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_agent(args)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
