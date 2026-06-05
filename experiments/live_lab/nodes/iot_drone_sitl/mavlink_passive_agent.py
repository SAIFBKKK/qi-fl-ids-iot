"""QI-FL-IDS-IoT project author: Saif Ben Fredj.
P16.16 Phase 2 Live Lab - 2026-06-05.
Continuous passive/synthetic MAVLink packet-window agent for the simulated UAV SITL node.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


REALTIME_DIR = Path(__file__).resolve().parents[2] / "realtime_agent"
sys.path.insert(0, str(REALTIME_DIR))

from flow_window import PacketWindow  # noqa: E402
from mqtt_runtime import publish_with_mosquitto_pub, publish_with_paho, utc_now  # noqa: E402
from scaler_runtime import apply_qga_mask, load_optional_scaler, load_qga_mask, scale_28_features  # noqa: E402


NODE_ID = "iot-drone-sitl"
INPUT_MODE = "selected_12_scaled"
MAVLINK_LISTEN_PORT = 14551
MAVLINK_PROTOCOL_NUMBER = 17.0
STATUS_TOPIC_KIND = "status"
WINDOW_TOPIC_KIND = "windows"
FEATURE_NAMES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "SSH",
    "TCP",
    "UDP",
    "ARP",
    "ICMP",
    "Tot sum",
    "Min",
    "Std",
    "IAT",
    "Number",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_scaler_path() -> Path:
    return repo_root() / "experiments" / "live_lab" / "artifacts" / "l1_binary_robust_scaler.json"


def packet(timestamp: float, length: int, source: str = "synthetic") -> dict[str, Any]:
    return {
        "timestamp": float(timestamp),
        "protocol": "UDP",
        "length": int(length),
        "src": source,
        "dst": NODE_ID,
        "flags": {},
        "sport": 14550,
        "dport": MAVLINK_LISTEN_PORT,
    }


def synthetic_packets(scenario: str, window_size: int) -> list[dict[str, Any]]:
    """Create packet metadata only; no network traffic is generated."""
    packets: list[dict[str, Any]] = []
    if scenario == "normal-sim":
        for index in range(window_size):
            packets.append(packet(timestamp=float(index), length=48 + (index % 5), source="normal-sim"))
        return packets
    if scenario == "burst-sim":
        for index in range(window_size):
            length = 48 + ((index * 7) % 25)
            packets.append(packet(timestamp=index * 0.002, length=length, source="burst-sim"))
        return packets
    raise ValueError(f"unsupported synthetic scenario: {scenario}")


def synthetic_packet_stream(scenario: str) -> Iterable[dict[str, Any]]:
    """Yield deterministic packet metadata forever without generating network traffic."""
    index = 0
    while True:
        if scenario == "normal-sim":
            yield packet(timestamp=float(index), length=48 + (index % 5), source="normal-sim")
        elif scenario == "burst-sim":
            yield packet(timestamp=index * 0.002, length=48 + ((index * 7) % 25), source="burst-sim")
        else:
            raise ValueError(f"unsupported synthetic scenario: {scenario}")
        index += 1


def passive_udp_packets(listen_port: int, window_size: int, timeout_seconds: int = 30) -> Iterable[dict[str, Any]]:
    """Listen for UDP metadata without generating traffic or storing application payloads."""
    deadline = time.time() + timeout_seconds
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("0.0.0.0", listen_port))
        server.settimeout(1.0)
        while time.time() < deadline:
            try:
                data, address = server.recvfrom(2048)
            except socket.timeout:
                continue
            yield {
                "timestamp": time.time(),
                "protocol": "UDP",
                "length": len(data),
                "src": address[0],
                "dst": "0.0.0.0",
                "flags": {},
                "sport": int(address[1]),
                "dport": listen_port,
            }
            if window_size <= 0:
                break


def passive_udp_stream(listen_port: int) -> Iterable[dict[str, Any]]:
    """Yield UDP packet metadata passively until interrupted; application payloads are not stored."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("0.0.0.0", listen_port))
        server.settimeout(1.0)
        while True:
            try:
                data, address = server.recvfrom(2048)
            except socket.timeout:
                continue
            yield {
                "timestamp": time.time(),
                "protocol": "UDP",
                "length": len(data),
                "src": address[0],
                "dst": "0.0.0.0",
                "flags": {},
                "sport": int(address[1]),
                "dport": listen_port,
            }


def window_from_packets(packets: Iterable[dict[str, Any]], window_size: int) -> PacketWindow:
    window = PacketWindow(window_size=window_size, flow_id=f"mavlink-window-{uuid4().hex[:12]}")
    for item in packets:
        window.add_packet(item)
        if window.is_ready():
            break
    return window


def timestamp_seconds(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_mavlink_features_28(window: PacketWindow) -> dict[str, Any]:
    """Map MAVLink/UDP packet metadata to the project 28-feature order.

    Scientific note: MAVLink SITL traffic is UDP. TCP flags such as SYN, ACK,
    FIN and RST are therefore zero here and must not be interpreted as MAVLink
    heartbeat counters. Burst-like MAVLink behavior is represented through
    Rate, UDP, IAT, Number, Tot sum and Std only.
    """
    packets = list(window.packets)
    timestamps = sorted(timestamp_seconds(item.get("timestamp")) for item in packets)
    lengths = [max(0, int(item.get("length", 0) or 0)) for item in packets]
    number = len(packets)
    flow_duration = max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
    rate = float(number / flow_duration) if flow_duration > 0 else float(number)
    iats = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    mean_iat = statistics.fmean(iats) if iats else 0.0
    length_std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    total_length = float(sum(lengths))
    minimum_length = float(min(lengths) if lengths else 0)
    average_length = float(statistics.fmean(lengths) if lengths else 0.0)

    # [0] flow_duration: elapsed time between the first and last packet.
    # [1] Header_Length: safe approximation using average observed UDP packet size.
    # [2] Protocol Type: constant 17 for UDP/IP.
    # [3] Duration: same window duration as flow_duration for this prototype.
    # [4] Rate: packets per second; key indicator for high-density UDP behavior.
    # [5:14] TCP flag/count features: zero because MAVLink here is UDP, not TCP.
    # [15:18] HTTP/HTTPS/DNS/SSH: zero because this agent observes MAVLink/UDP metadata.
    # [19] TCP: zero for MAVLink/UDP.
    # [20] UDP: number of UDP packets in the window; key protocol feature.
    # [21:22] ARP/ICMP: zero for this MAVLink observation path.
    # [23] Tot sum: total packet length in the window.
    # [24] Min: minimum packet length.
    # [25] Std: packet length population standard deviation.
    # [26] IAT: mean inter-arrival time; decreases in burst-like windows.
    # [27] Number: packet count in the PacketWindow.
    features_28 = [
        flow_duration,
        average_length,
        MAVLINK_PROTOCOL_NUMBER,
        flow_duration,
        rate,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(number),
        0.0,
        0.0,
        total_length,
        minimum_length,
        float(length_std),
        float(mean_iat),
        float(number),
    ]
    return {
        "feature_names": FEATURE_NAMES,
        "features_28": [float(value) for value in features_28],
        "approximated_features": [
            "Header_Length",
            "Protocol Type",
            "Duration",
            "Rate",
            "UDP",
            "Tot sum",
            "Min",
            "Std",
            "IAT",
            "Number",
        ],
        "unsupported_features": [],
        "scientific_notes": [
            "MAVLink SITL is represented as UDP metadata only.",
            "syn_flag_number and syn_count remain zero because UDP has no TCP SYN flag.",
            "burst-sim changes Rate, UDP, IAT, Number, Tot sum and Std without generating traffic.",
        ],
    }


def transform_features(features_28: list[float], scaler_path: str | Path | None = None) -> dict[str, Any]:
    scaler = load_optional_scaler(scaler_path)
    mask = load_qga_mask()
    scaled_28 = scale_28_features(features_28, scaler=scaler)
    selected_12 = apply_qga_mask(scaled_28)
    return {
        "features_28": [float(value) for value in features_28],
        "scaled_28": [float(value) for value in scaled_28],
        "selected_12": [float(value) for value in selected_12],
        "scaler": {**scaler.describe(), "used": scaler.available},
        "qga_mask": {
            "used": True,
            "selected_mask_id": mask["selected_mask_id"],
            "selected_indices": mask["selected_indices"],
            "warnings": mask["warnings"],
        },
    }


def build_payload(node_id: str, features_12: list[float], scenario: str, flow_id: str) -> dict[str, Any]:
    return {
        "flow_id": flow_id,
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": INPUT_MODE,
        "scenario": f"mavlink_{scenario}",
        "observation_mode": "passive_udp_metadata" if scenario == "passive" else "synthetic_packet_metadata",
        "protocol": "MAVLink/UDP",
        "mavlink_port": 14550,
        "features": [float(value) for value in features_12],
    }


def build_status_payload(
    *,
    node_id: str,
    windows_published: int,
    packets_received: int,
    last_window_id: str | None,
    uptime_seconds: float,
    errors: int,
    listen_port: int,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "timestamp": utc_now(),
        "event_type": "drone_observer_status",
        "agent_status": "running",
        "windows_published": int(windows_published),
        "packets_received": int(packets_received),
        "last_window_id": last_window_id,
        "last_prediction_label": None,
        "uptime_seconds": round(float(uptime_seconds), 3),
        "errors": int(errors),
        "protocol": "MAVLink/UDP",
        "listen_port": int(listen_port),
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
    features_28: list[float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_id": node_id,
        "timestamp": utc_now(),
        "event_type": "packet_window_update",
        "packet_count": int(packet_count),
        "buffer_fill": int(buffer_fill),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "last_window_id": last_window_id,
        "status": status,
    }
    if window_number is not None:
        payload["window_number"] = int(window_number)
    if features_28 is not None:
        payload.update(
            {
                "last_rate": float(features_28[4]),
                "last_iat": float(features_28[26]),
                "last_udp_count": float(features_28[20]),
                "last_number": float(features_28[27]),
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


def publish_event_payload(
    *,
    broker: str,
    port: int,
    topic: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return publish_payload(broker=broker, port=port, topic=topic, payload=payload, dry_run=dry_run)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger("mavlink_passive_agent")
    if args.scenario == "passive":
        logger.info("PASSIVE MODE - no traffic is generated by this agent")
        packets = passive_udp_packets(args.listen_port, args.window_size)
    elif args.scenario == "normal-sim":
        logger.info("MODE SIMULATION - aucun trafic reseau genere")
        packets = synthetic_packets("normal-sim", args.window_size)
    else:
        logger.info("MODE SIMULATION BURST - aucun trafic reseau genere - scenario synthetique uniquement")
        packets = synthetic_packets("burst-sim", args.window_size)

    window = window_from_packets(packets, args.window_size)
    if not window.is_ready():
        return {
            "ok": False,
            "node_id": args.node_id,
            "scenario": args.scenario,
            "window_summary": window.summary(),
            "warnings": ["No complete PacketWindow was produced."],
        }

    extraction = extract_mavlink_features_28(window)
    transform = transform_features(extraction["features_28"], args.scaler_path)
    topic = f"ids/flows/{args.node_id}"
    payload = build_payload(args.node_id, transform["selected_12"], args.scenario, window.flow_id)
    publish_result = publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=topic,
        payload=payload,
        dry_run=args.dry_run,
    )
    features = extraction["features_28"]
    logger.info(
        "[WINDOW] ts=%s node=%s mode=%s paquets=%s",
        utc_now(),
        args.node_id,
        args.scenario,
        int(features[27]),
    )
    logger.info(
        "[FEATURES-28] flow_duration=%.6f Rate=%.6f UDP=%.0f IAT=%.6f Number=%.0f",
        features[0],
        features[4],
        features[20],
        features[26],
        features[27],
    )
    logger.info("[FEATURES-12] %s", json.dumps(transform["selected_12"]))
    logger.info("[MQTT] topic=%s status=%s", topic, "dry-run" if args.dry_run else publish_result.get("published"))

    return {
        "ok": True,
        "node_id": args.node_id,
        "scenario": args.scenario,
        "input_mode": INPUT_MODE,
        "window_summary": window.summary(),
        "feature_names": FEATURE_NAMES,
        "features_28": extraction["features_28"],
        "features_12": transform["selected_12"],
        "transform": {
            "scaler": transform["scaler"],
            "qga_mask": transform["qga_mask"],
        },
        "mqtt": publish_result,
        "scientific_notes": extraction["scientific_notes"],
    }


def packet_source_for_continuous(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.scenario == "passive":
        logging.getLogger("mavlink_passive_agent").info("PASSIVE CONTINUOUS MODE - no traffic is generated by this agent")
        return passive_udp_stream(args.listen_port)
    logging.getLogger("mavlink_passive_agent").info(
        "CONTINUOUS SIMULATION MODE - aucun trafic reseau genere - scenario=%s",
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
    topic = f"ids/{STATUS_TOPIC_KIND}/{args.node_id}"
    payload = build_status_payload(
        node_id=args.node_id,
        windows_published=windows_published,
        packets_received=packets_received,
        last_window_id=last_window_id,
        uptime_seconds=time.time() - started_at,
        errors=errors,
        listen_port=args.listen_port,
    )
    return publish_event_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=topic,
        payload=payload,
        dry_run=args.dry_run,
    )


def publish_window_update(
    *,
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return publish_event_payload(
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
    window = PacketWindow(window_size=args.window_size, flow_id=f"mavlink-window-{uuid4().hex[:12]}")
    for item in list(buffer):
        window.add_packet(item)
    extraction = extract_mavlink_features_28(window)
    transform = transform_features(extraction["features_28"], args.scaler_path)
    flow_payload = build_payload(args.node_id, transform["selected_12"], args.scenario, window.flow_id)
    flow_publish = publish_payload(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=f"ids/flows/{args.node_id}",
        payload=flow_payload,
        dry_run=args.dry_run,
    )
    features = extraction["features_28"]
    window_payload = build_window_update_payload(
        node_id=args.node_id,
        status="ready",
        packet_count=args.window_size,
        buffer_fill=len(buffer),
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_number=window_number,
        last_window_id=window.flow_id,
        features_28=features,
    )
    window_publish = publish_window_update(args=args, payload=window_payload)
    log_row = {
        "ts": utc_now(),
        "window_number": window_number,
        "flow_id": window.flow_id,
        "scenario": args.scenario,
        "rate": float(features[4]),
        "iat": float(features[26]),
        "udp": float(features[20]),
        "number": float(features[27]),
        "features_12": transform["selected_12"],
        "predicted_label": None,
        "confidence": None,
    }
    write_jsonl(args.log_file, log_row)
    logging.getLogger("mavlink_passive_agent").info(
        "[WINDOW] continuous window=%s packet_total=%s flow_id=%s Rate=%.6f UDP=%.0f IAT=%.6f Number=%.0f",
        window_number,
        packets_received_total,
        window.flow_id,
        features[4],
        features[20],
        features[26],
        features[27],
    )
    return {
        "window_number": window_number,
        "packets_received_total": packets_received_total,
        "flow_id": window.flow_id,
        "feature_names": FEATURE_NAMES,
        "features_28": extraction["features_28"],
        "features_12": transform["selected_12"],
        "transform": {
            "scaler": transform["scaler"],
            "qga_mask": transform["qga_mask"],
        },
        "mqtt": flow_publish,
        "window_update": window_publish,
        "log_row": log_row,
    }


def run_continuous(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger("mavlink_passive_agent")
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
        logger.info("Continuous observer stopped by user.")
    except Exception as exc:  # noqa: BLE001 - runtime errors are surfaced in the result.
        errors += 1
        logger.warning("Continuous observer error: %s", exc)

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
            "MAVLink/UDP keeps syn_flag_number and syn_count at zero.",
            "Continuous synthetic modes generate packet metadata in memory only.",
            "Sliding windows use packets_received_total and last_publish_packet_count for stride correctness.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive/synthetic MAVLink PacketWindow agent for iot-drone-sitl.")
    parser.add_argument("--node-id", default=NODE_ID)
    parser.add_argument("--listen-port", type=int, default=MAVLINK_LISTEN_PORT)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--mqtt-broker", default="192.168.56.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--scaler-path", default=str(default_scaler_path()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenario", choices=("passive", "normal-sim", "burst-sim"), default="passive")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--status-interval", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.listen_port <= 0:
        parser.error("--listen-port must be positive")
    if args.window_stride <= 0:
        parser.error("--window-stride must be positive")
    if args.status_interval <= 0:
        parser.error("--status-interval must be positive")
    if args.max_windows < 0:
        parser.error("--max-windows must be zero or positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    result = run_continuous(args) if args.continuous else run_once(args)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
