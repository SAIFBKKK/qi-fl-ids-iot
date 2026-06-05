from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from feature_schema import FeatureSchema, load_schema
from nfstream_adapter import FlowStats


APP_PORTS = {
    "HTTP": {80, 8080},
    "HTTPS": {443, 8443},
    "DNS": {53},
    "SSH": {22},
}


@dataclass
class FeatureMappingResult:
    flow_id: str
    source: str
    feature_names: list[str]
    values: list[float | None]
    unsupported_features: list[str] = field(default_factory=list)
    approximated_features: dict[str, str] = field(default_factory=dict)
    source_fields: dict[str, list[str]] = field(default_factory=dict)

    def value_by_name(self) -> dict[str, float | None]:
        return dict(zip(self.feature_names, self.values, strict=True))


def _first(stats: dict[str, Any], *names: str) -> tuple[Any | None, str | None]:
    for name in names:
        value = stats.get(name)
        if value is not None and value != "":
            return value, name
    return None, None


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_protocol(stats: dict[str, Any]) -> tuple[float | None, list[str]]:
    value, field = _first(stats, "protocol", "ip_protocol", "protocol_number")
    numeric = _number(value)
    if numeric is not None:
        return numeric, [field] if field else []

    name, name_field = _first(stats, "protocol_name", "ip_protocol_name")
    if isinstance(name, str):
        normalized = name.upper()
        if normalized == "TCP":
            return 6.0, [name_field] if name_field else []
        if normalized == "UDP":
            return 17.0, [name_field] if name_field else []
        if normalized == "ICMP":
            return 1.0, [name_field] if name_field else []
        if normalized == "ARP":
            return 2054.0, [name_field] if name_field else []
    return None, []


def _duration_seconds(stats: dict[str, Any]) -> tuple[float | None, list[str], str | None]:
    value, field = _first(stats, "bidirectional_duration_ms", "duration_ms", "flow_duration_ms")
    numeric = _number(value)
    if numeric is not None:
        return numeric / 1000.0, [field] if field else [], "NFStream duration in ms converted to seconds."

    value, field = _first(stats, "bidirectional_duration", "duration", "flow_duration")
    numeric = _number(value)
    if numeric is not None:
        return numeric, [field] if field else [], None
    return None, [], None


def _packet_count(stats: dict[str, Any]) -> tuple[float | None, list[str]]:
    value, field = _first(stats, "bidirectional_packets", "packets", "packet_count", "Number")
    numeric = _number(value)
    return numeric, [field] if numeric is not None and field else []


def _flag_count(stats: dict[str, Any], flag: str) -> tuple[float | None, list[str]]:
    lower = flag.lower()
    value, field = _first(
        stats,
        f"bidirectional_{lower}_packets",
        f"{lower}_packets",
        f"{lower}_count",
        f"{lower}_flag_count",
        f"{lower}_flag_number",
    )
    numeric = _number(value)
    return numeric, [field] if numeric is not None and field else []


def _ports(stats: dict[str, Any]) -> set[int]:
    ports: set[int] = set()
    for name in ("src_port", "dst_port", "source_port", "destination_port", "server_port", "client_port"):
        numeric = _number(stats.get(name))
        if numeric is not None:
            ports.add(int(numeric))
    return ports


def _app_indicator(stats: dict[str, Any], app_name: str) -> tuple[float, list[str], str]:
    app_value, app_field = _first(stats, "application_name", "application_category_name", "requested_server_name")
    app_text = str(app_value or "").upper()
    ports = _ports(stats)
    if app_name in app_text or ports.intersection(APP_PORTS[app_name]):
        sources = [field for field in [app_field, "src_port", "dst_port"] if field]
        return 1.0, sources, "Application indicator inferred from NFStream application metadata and/or common ports."
    return 0.0, [field for field in [app_field] if field], "Application indicator inferred as absent from available metadata."


def map_nfstream_flow_to_ciciot_features(flow: FlowStats, schema: FeatureSchema | None = None) -> FeatureMappingResult:
    feature_schema = schema or load_schema()
    stats = flow.stats
    values_by_name: dict[str, float | None] = {}
    source_fields: dict[str, list[str]] = {}
    approximated: dict[str, str] = {}

    def set_value(name: str, value: float | None, sources: list[str] | None = None, approximation: str | None = None) -> None:
        values_by_name[name] = value
        if sources:
            source_fields[name] = sources
        if approximation and value is not None:
            approximated[name] = approximation

    duration, duration_sources, duration_approx = _duration_seconds(stats)
    set_value("flow_duration", duration, duration_sources, duration_approx)
    set_value("Duration", duration, duration_sources, duration_approx)

    header_value, header_field = _first(stats, "bidirectional_header_bytes", "header_bytes", "Header_Length")
    set_value(
        "Header_Length",
        _number(header_value),
        [header_field] if header_field else [],
        "Header length is available only when the adapter/source provides aggregate header bytes.",
    )

    protocol, protocol_sources = _int_protocol(stats)
    set_value("Protocol Type", protocol, protocol_sources)

    packet_count, packet_sources = _packet_count(stats)
    if packet_count is not None and duration is not None:
        safe_duration = duration if duration > 0 else 1e-9
        set_value("Rate", packet_count / safe_duration, packet_sources + duration_sources, "Rate derived as packet_count / duration.")
    else:
        set_value("Rate", None)

    for feature_name, flag in [
        ("fin_flag_number", "fin"),
        ("syn_flag_number", "syn"),
        ("rst_flag_number", "rst"),
        ("psh_flag_number", "psh"),
        ("ack_flag_number", "ack"),
        ("ack_count", "ack"),
        ("syn_count", "syn"),
        ("fin_count", "fin"),
        ("urg_count", "urg"),
        ("rst_count", "rst"),
    ]:
        value, sources = _flag_count(stats, flag)
        set_value(feature_name, value, sources, "Flag feature mapped from NFStream bidirectional packet flag counters.")

    for app_name in ("HTTP", "HTTPS", "DNS", "SSH"):
        value, sources, note = _app_indicator(stats, app_name)
        set_value(app_name, value, sources, note)

    if protocol is None:
        set_value("TCP", None)
        set_value("UDP", None)
        set_value("ARP", None)
        set_value("ICMP", None)
    else:
        set_value("TCP", 1.0 if int(protocol) == 6 else 0.0, protocol_sources)
        set_value("UDP", 1.0 if int(protocol) == 17 else 0.0, protocol_sources)
        set_value("ARP", 1.0 if int(protocol) == 2054 else 0.0, protocol_sources)
        set_value("ICMP", 1.0 if int(protocol) == 1 else 0.0, protocol_sources)

    total_bytes, total_field = _first(stats, "bidirectional_bytes", "bytes", "total_bytes", "Tot sum")
    set_value(
        "Tot sum",
        _number(total_bytes),
        [total_field] if total_field else [],
        "Mapped from NFStream bidirectional byte total; CICIoT packet-size semantics may differ.",
    )

    min_packet, min_field = _first(stats, "bidirectional_min_ps", "min_packet_size", "Min")
    set_value("Min", _number(min_packet), [min_field] if min_field else [])

    std_packet, std_field = _first(stats, "bidirectional_stddev_ps", "std_packet_size", "Std")
    set_value("Std", _number(std_packet), [std_field] if std_field else [])

    iat_value, iat_field = _first(stats, "bidirectional_mean_piat_ms", "mean_iat_ms", "IAT")
    iat_numeric = _number(iat_value)
    set_value(
        "IAT",
        iat_numeric / 1000.0 if iat_numeric is not None and iat_field and iat_field.endswith("_ms") else iat_numeric,
        [iat_field] if iat_field else [],
        "NFStream packet inter-arrival time in ms converted to seconds." if iat_field and iat_field.endswith("_ms") else None,
    )

    set_value("Number", packet_count, packet_sources)

    ordered = [values_by_name.get(name) for name in feature_schema.feature_names]
    unsupported = [name for name, value in zip(feature_schema.feature_names, ordered, strict=True) if value is None]
    return FeatureMappingResult(
        flow_id=flow.flow_id,
        source=flow.source,
        feature_names=feature_schema.feature_names,
        values=ordered,
        unsupported_features=unsupported,
        approximated_features=approximated,
        source_fields=source_fields,
    )


def map_flows(flows: list[FlowStats], schema: FeatureSchema | None = None) -> list[FeatureMappingResult]:
    feature_schema = schema or load_schema()
    return [map_nfstream_flow_to_ciciot_features(flow, feature_schema) for flow in flows]

