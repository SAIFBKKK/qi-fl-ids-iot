from __future__ import annotations

import ipaddress
from typing import Any

EXPECTED_FLAG_KEYS = ("syn", "ack", "psh", "fin", "rst", "urg")
ALLOWED_PROTOCOLS = {"tcp", "udp", "icmp", "arp"}


def _require_non_empty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{key}' must be a non-empty string")
    return value.strip()


def _require_ip_address(payload: dict[str, Any], key: str) -> str:
    value = _require_non_empty_string(payload, key)
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"field '{key}' must be a valid IPv4 or IPv6 address") from exc
    return value


def _require_number(payload: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"field '{key}' must be a number")
    numeric_value = float(value)
    if numeric_value < minimum:
        raise ValueError(f"field '{key}' must be >= {minimum}")
    return numeric_value


def _require_int(payload: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"field '{key}' must be an integer")
    if value < minimum:
        raise ValueError(f"field '{key}' must be >= {minimum}")
    return value


def _normalize_protocol(payload: dict[str, Any]) -> str:
    protocol = _require_non_empty_string(payload, "protocol").lower()
    if protocol not in ALLOWED_PROTOCOLS:
        allowed = ", ".join(sorted(ALLOWED_PROTOCOLS))
        raise ValueError(f"field 'protocol' must be one of: {allowed}")
    return protocol


def _infer_protocol_number(protocol: str) -> int:
    return {
        "tcp": 6,
        "udp": 17,
        "icmp": 1,
        "arp": 0,
    }[protocol]


def _infer_app_proto(protocol: str, src_port: int, dst_port: int) -> str:
    ports = {src_port, dst_port}
    if 80 in ports or 8080 in ports:
        return "http"
    if 443 in ports:
        return "https"
    if 53 in ports:
        return "dns"
    if 22 in ports:
        return "ssh"
    return protocol


def _normalize_flag_value(key: str, value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = int(value)
        if float(value) != float(numeric_value):
            raise ValueError(f"flag '{key}' must be 0 or 1")
        if numeric_value in (0, 1):
            return numeric_value
    raise ValueError(f"flag '{key}' must be 0 or 1")


def _normalize_flags(payload: dict[str, Any]) -> dict[str, int]:
    raw_flags = payload.get("flags")
    if not isinstance(raw_flags, dict):
        raise ValueError("field 'flags' must be an object")

    normalized: dict[str, int] = {}
    for key in EXPECTED_FLAG_KEYS:
        if key not in raw_flags:
            normalized[key] = 0
            continue
        normalized[key] = _normalize_flag_value(key, raw_flags[key])
    return normalized


def _normalize_flag_counts(payload: dict[str, Any], flags: dict[str, int], packet_count: int) -> dict[str, int]:
    raw_counts = payload.get("flag_counts")
    if not isinstance(raw_counts, dict):
        raise ValueError("field 'flag_counts' must be an object")

    normalized: dict[str, int] = {}
    for key in EXPECTED_FLAG_KEYS:
        if key not in raw_counts:
            normalized[key] = flags[key] * packet_count
            continue
        value = raw_counts[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"flag count '{key}' must be an integer >= 0")
        if value < 0:
            raise ValueError(f"flag count '{key}' must be an integer >= 0")
        normalized[key] = value
    return normalized


def _estimate_header_bytes_total(protocol: str, packet_count: int) -> int:
    per_packet = 40 if protocol == "tcp" else 28
    return packet_count * per_packet


def _estimate_defaults(event: dict[str, Any]) -> dict[str, Any]:
    packet_count = event["packet_count"]
    duration_ms = event["duration_ms"]
    duration_seconds = max(duration_ms / 1000.0, 1e-6)

    event["protocol_number"] = event.get("protocol_number", _infer_protocol_number(event["protocol"]))
    event["app_proto"] = event.get(
        "app_proto",
        _infer_app_proto(event["protocol"], event["src_port"], event["dst_port"]),
    )
    event["ttl"] = event.get("ttl", 64)
    event["header_bytes_total"] = event.get(
        "header_bytes_total",
        _estimate_header_bytes_total(event["protocol"], packet_count),
    )
    event["min_packet_size"] = event.get("min_packet_size", event["packet_size"])
    event["packet_size_std"] = event.get("packet_size_std", 0.0)
    event["iat_ns_mean"] = event.get(
        "iat_ns_mean",
        (duration_ms * 1_000_000) / max(packet_count, 1),
    )
    event["window_packet_mean"] = event.get("window_packet_mean", float(packet_count))
    event["request_rate"] = event.get(
        "request_rate",
        packet_count / duration_seconds,
    )
    return event


def validate_raw_event(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("raw event payload must be a dictionary")

    schema_version = _require_non_empty_string(payload, "schema_version")
    if schema_version != "1.0":
        raise ValueError("field 'schema_version' must be '1.0'")

    event_type = _require_non_empty_string(payload, "event_type")
    if event_type != "raw_iot_event":
        raise ValueError("field 'event_type' must be 'raw_iot_event'")

    validated: dict[str, Any] = {
        "schema_version": schema_version,
        "event_type": event_type,
        "timestamp": _require_non_empty_string(payload, "timestamp"),
        "node_id": _require_non_empty_string(payload, "node_id"),
        "gateway_id": _require_non_empty_string(payload, "gateway_id"),
        "node_group": _require_non_empty_string(payload, "node_group"),
        "device_type": _require_non_empty_string(payload, "device_type"),
        "src_ip": _require_ip_address(payload, "src_ip"),
        "dst_ip": _require_ip_address(payload, "dst_ip"),
        "src_port": _require_int(payload, "src_port", minimum=0),
        "dst_port": _require_int(payload, "dst_port", minimum=0),
        "protocol": _normalize_protocol(payload),
        "packet_size": _require_number(payload, "packet_size", minimum=0.0),
        "packet_count": _require_int(payload, "packet_count", minimum=1),
        "duration_ms": _require_number(payload, "duration_ms", minimum=0.0),
        "bytes_in": _require_number(payload, "bytes_in", minimum=0.0),
        "bytes_out": _require_number(payload, "bytes_out", minimum=0.0),
        "scenario": _require_non_empty_string(payload, "scenario"),
    }

    if validated["src_port"] > 65535 or validated["dst_port"] > 65535:
        raise ValueError("fields 'src_port' and 'dst_port' must be between 0 and 65535")

    flags = _normalize_flags(payload)
    flag_counts = _normalize_flag_counts(payload, flags, validated["packet_count"])
    validated["flags"] = flags
    validated["flag_counts"] = flag_counts

    optional_string_fields = ("app_proto", "simulated_label")
    for key in optional_string_fields:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"field '{key}' must be a non-empty string when provided")
        validated[key] = value.strip().lower() if key == "app_proto" else value.strip()

    optional_numeric_int_fields = {
        "protocol_number": 0,
        "ttl": 0,
        "header_bytes_total": 0,
    }
    for key, minimum in optional_numeric_int_fields.items():
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"field '{key}' must be an integer >= {minimum}")
        if value < minimum:
            raise ValueError(f"field '{key}' must be an integer >= {minimum}")
        validated[key] = value

    optional_numeric_fields = (
        "min_packet_size",
        "packet_size_std",
        "iat_ns_mean",
        "window_packet_mean",
        "request_rate",
    )
    for key in optional_numeric_fields:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"field '{key}' must be a number >= 0")
        numeric_value = float(value)
        if numeric_value < 0:
            raise ValueError(f"field '{key}' must be a number >= 0")
        validated[key] = numeric_value

    validated = _estimate_defaults(validated)
    return validated
