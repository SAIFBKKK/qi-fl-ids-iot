from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from flow_window import PacketWindow


FALLBACK_FEATURES = [
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

PROTOCOL_NUMBERS = {"ICMP": 1, "TCP": 6, "UDP": 17, "ARP": 2054, "OTHER": 0}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def feature_names_path() -> Path:
    return repo_root() / "experiments" / "qi-fl-ids-iot-final" / "outputs" / "artifacts" / "features" / "feature_names.json"


def load_feature_names(path: str | Path | None = None) -> list[str]:
    target = Path(path) if path else feature_names_path()
    if not target.exists():
        return list(FALLBACK_FEATURES)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"feature names file is not a list: {target}")
    names = [str(item) for item in value]
    if len(names) != 28:
        raise ValueError(f"expected 28 feature names, got {len(names)}")
    return names


def timestamp_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def packet_length(packet: dict[str, Any]) -> int:
    try:
        return max(0, int(packet.get("length", 0) or 0))
    except (TypeError, ValueError):
        return 0


def protocol(packet: dict[str, Any]) -> str:
    return str(packet.get("protocol", "OTHER") or "OTHER").upper()


def flag(packet: dict[str, Any], name: str) -> int:
    flags = packet.get("flags") or {}
    if not isinstance(flags, dict):
        return 0
    return int(bool(flags.get(name, False)))


def port_matches(packet: dict[str, Any], ports: set[int]) -> bool:
    return packet.get("sport") in ports or packet.get("dport") in ports


def approximate_header_length(packet: dict[str, Any]) -> int:
    item_protocol = protocol(packet)
    if item_protocol == "TCP":
        return 40
    if item_protocol in {"UDP", "ICMP"}:
        return 28
    if item_protocol == "ARP":
        return 28
    return 0


def protocol_type_value(protocol_counts: Counter[str]) -> float:
    if not protocol_counts:
        return 0.0
    dominant = protocol_counts.most_common(1)[0][0]
    return float(PROTOCOL_NUMBERS.get(dominant, 0))


def extract_28_features_from_window(window: PacketWindow) -> dict[str, Any]:
    packets = list(window.packets)
    feature_names = load_feature_names()
    timestamps = sorted(timestamp_seconds(packet.get("timestamp")) for packet in packets)
    lengths = [packet_length(packet) for packet in packets]
    protocol_counts = Counter(protocol(packet) for packet in packets)
    number = len(packets)
    duration = max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
    rate = float(number / duration) if duration > 0 else float(number)
    iats = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    mean_iat = statistics.fmean(iats) if iats else 0.0
    length_std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0

    values_by_name = {
        "flow_duration": duration,
        "Header_Length": float(sum(approximate_header_length(packet) for packet in packets)),
        "Protocol Type": protocol_type_value(protocol_counts),
        "Duration": duration,
        "Rate": rate,
        "fin_flag_number": float(sum(flag(packet, "fin") for packet in packets)),
        "syn_flag_number": float(sum(flag(packet, "syn") for packet in packets)),
        "rst_flag_number": float(sum(flag(packet, "rst") for packet in packets)),
        "psh_flag_number": float(sum(flag(packet, "psh") for packet in packets)),
        "ack_flag_number": float(sum(flag(packet, "ack") for packet in packets)),
        "ack_count": float(sum(flag(packet, "ack") for packet in packets)),
        "syn_count": float(sum(flag(packet, "syn") for packet in packets)),
        "fin_count": float(sum(flag(packet, "fin") for packet in packets)),
        "urg_count": float(sum(flag(packet, "urg") for packet in packets)),
        "rst_count": float(sum(flag(packet, "rst") for packet in packets)),
        "HTTP": float(sum(1 for packet in packets if port_matches(packet, {80, 8080}))),
        "HTTPS": float(sum(1 for packet in packets if port_matches(packet, {443}))),
        "DNS": float(sum(1 for packet in packets if port_matches(packet, {53}))),
        "SSH": float(sum(1 for packet in packets if port_matches(packet, {22}))),
        "TCP": float(protocol_counts.get("TCP", 0)),
        "UDP": float(protocol_counts.get("UDP", 0)),
        "ARP": float(protocol_counts.get("ARP", 0)),
        "ICMP": float(protocol_counts.get("ICMP", 0)),
        "Tot sum": float(sum(lengths)),
        "Min": float(min(lengths) if lengths else 0),
        "Std": float(length_std),
        "IAT": float(mean_iat),
        "Number": float(number),
    }
    features_28 = [float(values_by_name.get(name, 0.0)) for name in feature_names]
    return {
        "features_28": features_28,
        "feature_names": feature_names,
        "approximated_features": list(feature_names),
        "unsupported_features": [],
        "notes": [
            "P16.7 computes safe CICIoT2023-like approximations from packet windows.",
            "These values are not claimed to be equivalent to the original CICIoT2023 extractor.",
            "Application protocol counters are inferred from source or destination ports.",
        ],
    }
