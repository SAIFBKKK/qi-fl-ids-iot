"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
P16.17 - Continuous Smartwatch Traffic Observer.
Passive/synthetic PacketWindow(30) observer for iot-smart-watch-medium.

This agent never generates network traffic. Passive mode observes packet
metadata only, and simulation modes create local packet metadata in memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


REALTIME_DIR = Path(__file__).resolve().parents[2] / "realtime_agent"
sys.path.insert(0, str(REALTIME_DIR))

from feature_extractor import FALLBACK_FEATURES, extract_28_features_from_window  # noqa: E402
from flow_window import PacketWindow  # noqa: E402
from mqtt_runtime import publish_with_mosquitto_pub, publish_with_paho, utc_now  # noqa: E402
from passive_capture import passive_packet_source  # noqa: E402
from scaler_runtime import load_optional_scaler, scale_28_features  # noqa: E402


NODE_ID = "iot-smart-watch-medium"
DEVICE_TYPE = "smart_watch"
PROTOCOL_FOCUS = "ICMP/TCP/HTTP-like"
INPUT_MODE = "original_28_scaled"
DEFAULT_INTERFACE = "enp0s8"
DEFAULT_NODE_IP = "192.168.56.102"
DEFAULT_PEER_IP = "192.168.56.103"
DEFAULT_BROKER = "192.168.56.1"
STATUS_TOPIC_KIND = "status"
WINDOW_TOPIC_KIND = "windows"
FEATURE_NAMES = list(FALLBACK_FEATURES)

RATE_INDEX = FEATURE_NAMES.index("Rate")
IAT_INDEX = FEATURE_NAMES.index("IAT")
SYN_FLAG_INDEX = FEATURE_NAMES.index("syn_flag_number")
SYN_COUNT_INDEX = FEATURE_NAMES.index("syn_count")
TCP_INDEX = FEATURE_NAMES.index("TCP")
UDP_INDEX = FEATURE_NAMES.index("UDP")
ICMP_INDEX = FEATURE_NAMES.index("ICMP")
NUMBER_INDEX = FEATURE_NAMES.index("Number")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_scaler_path() -> Path:
    return repo_root() / "experiments" / "live_lab" / "artifacts" / "l1_binary_robust_scaler.json"


def packet(timestamp: float, length: int, source: str, *, protocol: str = "ICMP") -> dict[str, Any]:
    return {
        "timestamp": float(timestamp),
        "protocol": protocol,
        "length": int(length),
        "src": source,
        "dst": NODE_ID,
        "src_ip": DEFAULT_PEER_IP,
        "dst_ip": DEFAULT_NODE_IP,
        "flags": {},
        "sport": None,
        "dport": None,
        "src_port": None,
        "dst_port": None,
    }


def synthetic_packets(scenario: str, window_size: int) -> list[dict[str, Any]]:
    """Create packet metadata only; no network traffic is generated."""
    if scenario == "icmp-lowrate-sim":
        return [
            packet(timestamp=float(index), length=84 + (index % 3), source="icmp-lowrate-sim")
            for index in range(window_size)
        ]
    if scenario == "icmp-flood-sim":
        return [
            packet(timestamp=index * 0.003, length=84 + ((index * 5) % 17), source="icmp-flood-sim")
            for index in range(window_size)
        ]
    raise ValueError(f"unsupported synthetic scenario: {scenario}")


def synthetic_packet_stream(scenario: str) -> Iterable[dict[str, Any]]:
    """Yield deterministic packet metadata forever without generating network traffic."""
    index = 0
    while True:
        if scenario == "icmp-lowrate-sim":
            yield packet(timestamp=float(index), length=84 + (index % 3), source=scenario)
        elif scenario == "icmp-flood-sim":
            yield packet(timestamp=index * 0.003, length=84 + ((index * 5) % 17), source=scenario)
        else:
            raise ValueError(f"unsupported synthetic scenario: {scenario}")
        index += 1


def passive_packet_stream(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Yield passive packets in repeated safe host-only capture windows."""
    logger = logging.getLogger("smartwatch_passive_agent")
    while True:
        source = passive_packet_source(
            interface_name=args.interface,
            max_packets=max(args.window_stride, 1),
            timeout_seconds=args.timeout_seconds,
            node_ip=args.node_ip,
            peer_ip=args.peer_ip,
        )
        yielded = 0
        for item in source.iter_packets(max_packets=max(args.window_stride, 1)):
            yielded += 1
            yield item
        if yielded == 0:
            logger.info("Passive observer timeout; waiting for host-only traffic metadata.")


def window_from_packets(packets: Iterable[dict[str, Any]], window_size: int) -> PacketWindow:
    window = PacketWindow(window_size=window_size, flow_id=f"smartwatch-window-{uuid4().hex[:12]}")
    for item in packets:
        window.add_packet(item)
        if window.is_ready():
            break
    return window


def extract_smartwatch_features_28(window: PacketWindow) -> dict[str, Any]:
    extraction = extract_28_features_from_window(window)
    return {
        **extraction,
        "scientific_notes": [
            "Smartwatch ICMP observation uses packet metadata only.",
            "For ICMP windows, TCP, UDP, syn_flag_number and syn_count remain zero.",
            "Rate and IAT are CICIoT2023-like approximations, then normalized by the runtime RobustScaler JSON.",
        ],
    }


def transform_features_28(
    features_28: list[float],
    *,
    input_mode: str,
    scaler_path: str | Path | None = None,
) -> dict[str, Any]:
    scaler = load_optional_scaler(scaler_path)
    no_scale = input_mode == "original_28_unscaled"
    scaled_28 = scale_28_features(features_28, scaler=scaler, no_scale=no_scale)
    return {
        "features_28": [float(value) for value in features_28],
        "scaled_28": [float(value) for value in scaled_28],
        "output_features": [float(value) for value in (features_28 if no_scale else scaled_28)],
        "input_mode": input_mode,
        "scaler": {**scaler.describe(), "used": bool(scaler.available and not no_scale)},
        "qga_mask": {
            "used": False,
            "selected_mask_id": "conservative_seed_42",
            "reason": "VM2 publishes original_28_scaled; final-ids-api applies the QGA mask.",
        },
    }


def build_flow_payload(
    *,
    node_id: str,
    features: list[float],
    scenario: str,
    flow_id: str,
    input_mode: str,
) -> dict[str, Any]:
    return {
        "flow_id": flow_id,
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "scenario": f"smartwatch_{scenario}",
        "observation_mode": "real_traffic_passive_capture" if scenario == "passive" else "synthetic_packet_metadata",
        "protocol_focus": PROTOCOL_FOCUS,
        "features": [float(value) for value in features],
    }


def build_status_payload(
    *,
    node_id: str,
    windows_published: int,
    packets_received: int,
    last_window_id: str | None,
    uptime_seconds: float,
    errors: int,
    interface: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "timestamp": utc_now(),
        "event_type": "smartwatch_observer_status",
        "agent_status": "running",
        "windows_published": int(windows_published),
        "packets_received": int(packets_received),
        "last_window_id": last_window_id,
        "last_prediction_label": None,
        "uptime_seconds": round(float(uptime_seconds), 3),
        "errors": int(errors),
        "protocol_focus": PROTOCOL_FOCUS,
        "interface": interface,
    }


def build_window_update_payload(
    *,
    node_id: str,
    status: str,
    packet_count: int,
    buffer_fill: int,
    window_size: int,
    window_stride: int,
    window_number: int | None = None,
    last_window_id: str | None = None,
    scaled_28: list[float] | None = None,
    raw_28: list[float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_id": node_id,
        "timestamp": utc_now(),
        "event_type": "smartwatch_packet_window_update",
        "packet_count": int(packet_count),
        "buffer_fill": int(buffer_fill),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "last_window_id": last_window_id,
        "status": status,
    }
    if window_number is not None:
        payload["window_number"] = int(window_number)
    if scaled_28 is not None and raw_28 is not None:
        payload.update(
            {
                "last_rate_scaled": float(scaled_28[RATE_INDEX]),
                "last_iat_scaled": float(scaled_28[IAT_INDEX]),
                "last_icmp_count": float(raw_28[ICMP_INDEX]),
                "last_tcp_count": float(raw_28[TCP_INDEX]),
                "last_udp_count": float(raw_28[UDP_INDEX]),
                "last_number": float(raw_28[NUMBER_INDEX]),
            }
        )
    return payload


def write_jsonl(log_file: str | Path | None, row: dict[str, Any]) -> None:
    if not log_file:
        return
    target = Path(log_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def publish_payload(
    *,
    broker: str,
    port: int,
    topic: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {"topic": topic, "published": False, "dry_run": True, "payload": payload}
    try:
        publish_with_paho(
            broker=broker,
            port=port,
            username="ids_user",
            password="changeme_in_dotenv",
            topic=topic,
            payload=payload,
        )
        return {"topic": topic, "published": True, "publisher": "paho-mqtt", "payload": payload}
    except Exception as paho_error:
        try:
            publish_with_mosquitto_pub(
                broker=broker,
                port=port,
                username="ids_user",
                password="changeme_in_dotenv",
                topic=topic,
                payload=payload,
            )
            return {"topic": topic, "published": True, "publisher": "mosquitto_pub", "payload": payload}
        except Exception as fallback_error:
            return {"topic": topic, "published": False, "error": f"{paho_error}; {fallback_error}", "payload": payload}


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger("smartwatch_passive_agent")
    if args.scenario == "passive":
        logger.info("PASSIVE MODE - no traffic is generated by this agent")
        source = passive_packet_source(
            interface_name=args.interface,
            max_packets=args.window_size,
            timeout_seconds=args.timeout_seconds,
            node_ip=args.node_ip,
            peer_ip=args.peer_ip,
        )
        packets = source.iter_packets(max_packets=args.window_size)
    else:
        logger.info("SIMULATION MODE - metadata only - no network traffic generated")
        packets = synthetic_packets(args.scenario, args.window_size)

    window = window_from_packets(packets, args.window_size)
    if not window.is_ready():
        return {
            "ok": False,
            "node_id": args.node_id,
            "scenario": args.scenario,
            "window_summary": window.summary(),
            "warnings": ["No complete PacketWindow was produced."],
        }

    extraction = extract_smartwatch_features_28(window)
    transform = transform_features_28(
        extraction["features_28"],
        input_mode=args.input_mode,
        scaler_path=args.scaler_path,
    )
    topic = f"ids/flows/{args.node_id}"
    flow_payload = build_flow_payload(
        node_id=args.node_id,
        features=transform["output_features"],
        scenario=args.scenario,
        flow_id=window.flow_id,
        input_mode=args.input_mode,
    )
    publish_result = publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=topic,
        payload=flow_payload,
        dry_run=args.dry_run,
    )
    raw = extraction["features_28"]
    scaled = transform["scaled_28"]
    logger.info(
        "[WINDOW] node=%s scenario=%s packets=%s flow_id=%s",
        args.node_id,
        args.scenario,
        int(raw[NUMBER_INDEX]),
        window.flow_id,
    )
    logger.info(
        "[FEATURES] Rate_raw=%.6f Rate_scaled=%.6f ICMP=%.0f TCP=%.0f UDP=%.0f IAT_scaled=%.6f Number=%.0f",
        raw[RATE_INDEX],
        scaled[RATE_INDEX],
        raw[ICMP_INDEX],
        raw[TCP_INDEX],
        raw[UDP_INDEX],
        scaled[IAT_INDEX],
        raw[NUMBER_INDEX],
    )
    return {
        "ok": True,
        "node_id": args.node_id,
        "scenario": args.scenario,
        "input_mode": args.input_mode,
        "window_summary": window.summary(),
        "feature_names": extraction["feature_names"],
        "features_28": extraction["features_28"],
        "features_28_scaled": transform["scaled_28"],
        "output_features": transform["output_features"],
        "features_count": len(transform["output_features"]),
        "transform": {"scaler": transform["scaler"], "qga_mask": transform["qga_mask"]},
        "mqtt": publish_result,
        "scientific_notes": extraction["scientific_notes"],
    }


def packet_source_for_continuous(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.scenario == "passive":
        logging.getLogger("smartwatch_passive_agent").info(
            "PASSIVE CONTINUOUS MODE - passive observation only; no traffic is generated by this agent."
        )
        return passive_packet_stream(args)
    logging.getLogger("smartwatch_passive_agent").info(
        "CONTINUOUS SIMULATION MODE - metadata only - scenario=%s",
        args.scenario,
    )
    return synthetic_packet_stream(args.scenario)


def publish_status(
    *,
    args: argparse.Namespace,
    windows_published: int,
    packets_received: int,
    last_window_id: str | None,
    started_at: float,
    errors: int,
) -> dict[str, Any]:
    payload = build_status_payload(
        node_id=args.node_id,
        windows_published=windows_published,
        packets_received=packets_received,
        last_window_id=last_window_id,
        uptime_seconds=time.time() - started_at,
        errors=errors,
        interface=args.interface,
    )
    return publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=f"ids/{STATUS_TOPIC_KIND}/{args.node_id}",
        payload=payload,
        dry_run=args.dry_run,
    )


def publish_window_update(*, args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    return publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=f"ids/{WINDOW_TOPIC_KIND}/{args.node_id}",
        payload=payload,
        dry_run=args.dry_run,
    )


def process_completed_window(
    *,
    args: argparse.Namespace,
    buffer: deque[dict[str, Any]],
    window_number: int,
    packets_received_total: int,
) -> dict[str, Any]:
    window = PacketWindow(window_size=args.window_size, flow_id=f"smartwatch-window-{uuid4().hex[:12]}")
    for item in list(buffer):
        window.add_packet(item)
    extraction = extract_smartwatch_features_28(window)
    transform = transform_features_28(
        extraction["features_28"],
        input_mode=args.input_mode,
        scaler_path=args.scaler_path,
    )
    raw = extraction["features_28"]
    scaled = transform["scaled_28"]
    flow_payload = build_flow_payload(
        node_id=args.node_id,
        features=transform["output_features"],
        scenario=args.scenario,
        flow_id=window.flow_id,
        input_mode=args.input_mode,
    )
    flow_publish = publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=f"ids/flows/{args.node_id}",
        payload=flow_payload,
        dry_run=args.dry_run,
    )
    window_payload = build_window_update_payload(
        node_id=args.node_id,
        status="ready",
        packet_count=args.window_size,
        buffer_fill=len(buffer),
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_number=window_number,
        last_window_id=window.flow_id,
        scaled_28=scaled,
        raw_28=raw,
    )
    window_publish = publish_window_update(args=args, payload=window_payload)
    log_row = {
        "ts": utc_now(),
        "window_number": int(window_number),
        "flow_id": window.flow_id,
        "scenario": args.scenario,
        "protocol": "ICMP" if raw[ICMP_INDEX] >= raw[TCP_INDEX] and raw[ICMP_INDEX] >= raw[UDP_INDEX] else "MIXED",
        "rate_scaled": float(scaled[RATE_INDEX]),
        "iat_scaled": float(scaled[IAT_INDEX]),
        "icmp": float(raw[ICMP_INDEX]),
        "tcp": float(raw[TCP_INDEX]),
        "udp": float(raw[UDP_INDEX]),
        "number": float(raw[NUMBER_INDEX]),
        "features_28": transform["output_features"],
        "predicted_label": None,
        "confidence": None,
    }
    write_jsonl(args.log_file, log_row)
    logging.getLogger("smartwatch_passive_agent").info(
        "[WINDOW] continuous window=%s packet_total=%s flow_id=%s Rate_scaled=%.6f ICMP=%.0f TCP=%.0f UDP=%.0f IAT_scaled=%.6f Number=%.0f",
        window_number,
        packets_received_total,
        window.flow_id,
        scaled[RATE_INDEX],
        raw[ICMP_INDEX],
        raw[TCP_INDEX],
        raw[UDP_INDEX],
        scaled[IAT_INDEX],
        raw[NUMBER_INDEX],
    )
    return {
        "window_number": window_number,
        "packets_received_total": packets_received_total,
        "flow_id": window.flow_id,
        "feature_names": extraction["feature_names"],
        "features_28": raw,
        "features_28_scaled": scaled,
        "output_features": transform["output_features"],
        "features_count": len(transform["output_features"]),
        "transform": {"scaler": transform["scaler"], "qga_mask": transform["qga_mask"]},
        "mqtt": flow_publish,
        "window_update": window_publish,
        "log_row": log_row,
    }


def run_continuous(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger("smartwatch_passive_agent")
    buffer: deque[dict[str, Any]] = deque(maxlen=args.window_size)
    packets_received_total = 0
    last_publish_packet_count = 0
    windows_published = 0
    last_window_id: str | None = None
    errors = 0
    started_at = time.time()
    last_status_at = 0.0
    windows: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []
    window_events: list[dict[str, Any]] = []

    status_events.append(
        publish_status(
            args=args,
            windows_published=windows_published,
            packets_received=packets_received_total,
            last_window_id=last_window_id,
            started_at=started_at,
            errors=errors,
        )
    )
    last_status_at = time.time()

    try:
        for item in packet_source_for_continuous(args):
            packets_received_total += 1
            buffer.append(item)

            if len(buffer) < args.window_size:
                window_events.append(
                    publish_window_update(
                        args=args,
                        payload=build_window_update_payload(
                            node_id=args.node_id,
                            status="filling",
                            packet_count=packets_received_total,
                            buffer_fill=len(buffer),
                            window_size=args.window_size,
                            window_stride=args.window_stride,
                            last_window_id=last_window_id,
                        ),
                    )
                )

            should_publish = (
                len(buffer) >= args.window_size
                and packets_received_total - last_publish_packet_count >= args.window_stride
            )
            if should_publish:
                windows_published += 1
                window_result = process_completed_window(
                    args=args,
                    buffer=buffer,
                    window_number=windows_published,
                    packets_received_total=packets_received_total,
                )
                windows.append(window_result)
                window_events.append(window_result["window_update"])
                last_publish_packet_count = packets_received_total
                last_window_id = str(window_result["flow_id"])
                status_events.append(
                    publish_status(
                        args=args,
                        windows_published=windows_published,
                        packets_received=packets_received_total,
                        last_window_id=last_window_id,
                        started_at=started_at,
                        errors=errors,
                    )
                )
                last_status_at = time.time()
                if args.max_windows and windows_published >= args.max_windows:
                    break

            now = time.time()
            if now - last_status_at >= args.status_interval:
                status_events.append(
                    publish_status(
                        args=args,
                        windows_published=windows_published,
                        packets_received=packets_received_total,
                        last_window_id=last_window_id,
                        started_at=started_at,
                        errors=errors,
                    )
                )
                last_status_at = now
    except KeyboardInterrupt:
        logger.info("Continuous smartwatch observer stopped by user.")
    except Exception as exc:  # noqa: BLE001 - runtime errors are surfaced in the result.
        errors += 1
        logger.warning("Continuous smartwatch observer error: %s", exc)

    return {
        "ok": errors == 0 and (bool(windows) or args.max_windows == 0),
        "node_id": args.node_id,
        "scenario": args.scenario,
        "continuous": True,
        "dry_run": args.dry_run,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "max_windows": args.max_windows,
        "windows_published": windows_published,
        "packets_received_total": packets_received_total,
        "last_window_id": last_window_id,
        "status_events": status_events,
        "window_events": window_events,
        "windows": windows,
        "errors": errors,
        "log_file": args.log_file,
        "scientific_notes": [
            "ICMP windows keep TCP, UDP, syn_flag_number and syn_count at zero.",
            "Continuous synthetic modes generate packet metadata in memory only.",
            "Passive mode observes only host-only packet metadata and stores no application payload.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive/synthetic PacketWindow observer for iot-smart-watch-medium.")
    parser.add_argument("--node-id", default=NODE_ID)
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--peer-ip", default=DEFAULT_PEER_IP)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--mqtt-broker", default=DEFAULT_BROKER)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--input-mode", choices=("original_28_scaled", "original_28_unscaled"), default=INPUT_MODE)
    parser.add_argument("--scaler-path", default=str(default_scaler_path()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--status-interval", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--allow-live-capture", action="store_true")
    parser.add_argument("--scenario", choices=("passive", "icmp-lowrate-sim", "icmp-flood-sim"), default="passive")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args(argv)
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.window_stride <= 0:
        parser.error("--window-stride must be positive")
    if args.status_interval <= 0:
        parser.error("--status-interval must be positive")
    if args.max_windows < 0:
        parser.error("--max-windows must be zero or positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.scenario == "passive" and not args.allow_live_capture:
        parser.error("passive live observation requires --allow-live-capture")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    result = run_continuous(args) if args.continuous else run_once(args)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
