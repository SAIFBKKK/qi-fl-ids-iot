from __future__ import annotations

import math
from typing import Any

from raw_schema import validate_raw_event


CANONICAL_FEATURE_NAMES = [
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


def get_canonical_feature_names() -> list[str]:
    return list(CANONICAL_FEATURE_NAMES)


def _require_float(features: dict[str, Any], key: str) -> float:
    if key not in features:
        raise ValueError(f"mapped feature '{key}' is missing")
    value = features[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"mapped feature '{key}' must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"mapped feature '{key}' must be finite")
    return numeric_value


def _protocol_indicator(protocol: str, target: str) -> float:
    return 1.0 if protocol == target else 0.0


def _app_proto_indicator(app_proto: str, target: str) -> float:
    return 1.0 if app_proto == target else 0.0


def map_raw_to_features(raw_event: dict[str, Any]) -> dict[str, float]:
    """Map a raw IoT event to the canonical 28-feature CIC-IoT-compatible vector."""
    event = validate_raw_event(raw_event)
    protocol = event["protocol"]
    app_proto = event["app_proto"]
    flags = event["flags"]
    flag_counts = event["flag_counts"]

    mapped: dict[str, Any] = {
        "flow_duration": event["duration_ms"] / 1000.0,
        "Header_Length": event["header_bytes_total"],
        "Protocol Type": event["protocol_number"],
        "Duration": event["ttl"],
        "Rate": event["request_rate"],
        "fin_flag_number": flags["fin"],
        "syn_flag_number": flags["syn"],
        "rst_flag_number": flags["rst"],
        "psh_flag_number": flags["psh"],
        "ack_flag_number": flags["ack"],
        "ack_count": flag_counts["ack"],
        "syn_count": flag_counts["syn"],
        "fin_count": flag_counts["fin"],
        "urg_count": flag_counts["urg"],
        "rst_count": flag_counts["rst"],
        "HTTP": _app_proto_indicator(app_proto, "http"),
        "HTTPS": _app_proto_indicator(app_proto, "https"),
        "DNS": _app_proto_indicator(app_proto, "dns"),
        "SSH": _app_proto_indicator(app_proto, "ssh"),
        "TCP": _protocol_indicator(protocol, "tcp"),
        "UDP": _protocol_indicator(protocol, "udp"),
        "ARP": _protocol_indicator(protocol, "arp"),
        "ICMP": _protocol_indicator(protocol, "icmp"),
        "Tot sum": event["bytes_in"] + event["bytes_out"],
        "Min": event["min_packet_size"],
        "Std": event["packet_size_std"],
        "IAT": event["iat_ns_mean"],
        "Number": event["window_packet_mean"],
    }

    if set(mapped.keys()) != set(CANONICAL_FEATURE_NAMES):
        missing = sorted(set(CANONICAL_FEATURE_NAMES) - set(mapped.keys()))
        unexpected = sorted(set(mapped.keys()) - set(CANONICAL_FEATURE_NAMES))
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"mapped features do not match canonical schema: {', '.join(details)}")

    normalized: dict[str, float] = {}
    for feature_name in CANONICAL_FEATURE_NAMES:
        normalized[feature_name] = _require_float(mapped, feature_name)
    return normalized


def features_to_ordered_list(features: dict[str, float]) -> list[float]:
    return [_require_float(features, feature_name) for feature_name in CANONICAL_FEATURE_NAMES]
