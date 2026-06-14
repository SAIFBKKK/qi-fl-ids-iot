from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FlowStats:
    flow_id: str
    stats: dict[str, Any]
    source: str = "nfstream"


def nfstream_available() -> bool:
    try:
        import nfstream  # noqa: F401
    except ImportError:
        return False
    return True


def _public_flow_dict(flow: Any) -> dict[str, Any]:
    if isinstance(flow, dict):
        return dict(flow)
    if hasattr(flow, "to_dict"):
        value = flow.to_dict()
        if isinstance(value, dict):
            return value
    if hasattr(flow, "__dict__"):
        return {key: value for key, value in vars(flow).items() if not key.startswith("_")}

    result: dict[str, Any] = {}
    for name in dir(flow):
        if name.startswith("_"):
            continue
        try:
            value = getattr(flow, name)
        except Exception:  # noqa: BLE001 - best-effort adapter around external flow objects.
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            result[name] = value
    return result


def _flow_id(index: int, stats: dict[str, Any]) -> str:
    existing = stats.get("id") or stats.get("flow_id")
    if existing:
        return str(existing)
    src_ip = stats.get("src_ip") or stats.get("src_address") or "src"
    dst_ip = stats.get("dst_ip") or stats.get("dst_address") or "dst"
    src_port = stats.get("src_port") or 0
    dst_port = stats.get("dst_port") or 0
    protocol = stats.get("protocol") or stats.get("protocol_name") or "proto"
    return f"nfstream-{index:06d}-{src_ip}-{src_port}-{dst_ip}-{dst_port}-{protocol}"


def extract_flows_from_pcap(pcap_path: str | Path, limit: int | None = None) -> list[FlowStats]:
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"pcap file not found: {path}")
    try:
        from nfstream import NFStreamer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("NFStream is required to read pcap files; install requirements.txt first") from exc

    streamer = NFStreamer(source=str(path), statistical_analysis=True)
    flows: list[FlowStats] = []
    for index, flow in enumerate(streamer, start=1):
        stats = _public_flow_dict(flow)
        flows.append(FlowStats(flow_id=_flow_id(index, stats), stats=stats))
        if limit is not None and len(flows) >= limit:
            break
    return flows


def dry_run_flow_stats() -> list[FlowStats]:
    """Controlled in-memory NFStream-like rows used for schema tests without a pcap."""
    return [
        FlowStats(
            flow_id="dry-nfstream-flow-000001",
            source="dry_nfstream_like",
            stats={
                "src_ip": "10.16.0.10",
                "dst_ip": "10.16.0.20",
                "src_port": 51514,
                "dst_port": 443,
                "protocol": 6,
                "application_name": "TLS",
                "bidirectional_duration_ms": 1200.0,
                "bidirectional_packets": 10,
                "bidirectional_bytes": 5400,
                "bidirectional_min_ps": 60,
                "bidirectional_stddev_ps": 18.5,
                "bidirectional_mean_piat_ms": 120.0,
                "bidirectional_header_bytes": 400,
                "bidirectional_fin_packets": 1,
                "bidirectional_syn_packets": 1,
                "bidirectional_rst_packets": 0,
                "bidirectional_psh_packets": 2,
                "bidirectional_ack_packets": 8,
                "bidirectional_urg_packets": 0,
            },
        )
    ]


def flow_stats_from_dicts(rows: Iterable[dict[str, Any]]) -> list[FlowStats]:
    return [FlowStats(flow_id=_flow_id(index, row), stats=dict(row), source="dict") for index, row in enumerate(rows, start=1)]

