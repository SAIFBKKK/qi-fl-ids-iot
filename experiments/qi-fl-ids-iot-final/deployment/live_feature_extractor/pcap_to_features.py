from __future__ import annotations

import argparse
import csv
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from feature_schema import align_feature_row, schema_summary, selected_from_original
from flow_window import PacketObservation, completed_windows
from scaler_adapter import apply_scaler


APP_PORTS = {
    80: "HTTP",
    443: "HTTPS",
    53: "DNS",
    22: "SSH",
}


@dataclass
class ExtractionRecord:
    flow_id: str
    input_mode: str
    features: list[float | None]
    unsupported_features: list[str] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)

    def as_api_payload(self, node_id: str = "pcap-extractor") -> dict[str, object]:
        return {
            "flow_id": self.flow_id,
            "node_id": node_id,
            "input_mode": self.input_mode,
            "features": self.features,
            "unsupported_features": self.unsupported_features,
            "approximations": self.approximations,
        }


@dataclass
class ExtractionResult:
    records: list[ExtractionRecord]
    gaps: list[str]
    scaled: bool = False

    def to_payloads(self, node_id: str = "pcap-extractor") -> list[dict[str, object]]:
        return [record.as_api_payload(node_id=node_id) for record in self.records]


def inet_to_str(raw: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, raw)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, raw)


def app_protocol(src_port: int, dst_port: int) -> str | None:
    return APP_PORTS.get(src_port) or APP_PORTS.get(dst_port)


def packet_observations_from_pcap(path: str | Path) -> Iterable[PacketObservation]:
    try:
        import dpkt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dpkt is required for passive local pcap extraction") from exc

    with Path(path).open("rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        for timestamp, buffer in reader:
            try:
                ethernet = dpkt.ethernet.Ethernet(buffer)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            if isinstance(ethernet.data, dpkt.arp.ARP):
                src = inet_to_str(ethernet.data.spa)
                dst = inet_to_str(ethernet.data.tpa)
                yield PacketObservation(
                    timestamp=float(timestamp),
                    flow_key=(src, dst, 0, 0, "ARP"),
                    length=len(buffer),
                    header_length=28,
                    protocol_number=0,
                    app_protocol=None,
                    is_arp=1,
                )
                continue

            ip = ethernet.data
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue
            src = inet_to_str(ip.src)
            dst = inet_to_str(ip.dst)
            proto = int(getattr(ip, "p", getattr(ip, "nxt", 0)))
            transport = ip.data
            header_length = int(getattr(ip, "hl", 5)) * 4 if hasattr(ip, "hl") else 40

            if isinstance(transport, dpkt.tcp.TCP):
                src_port = int(transport.sport)
                dst_port = int(transport.dport)
                flags = int(transport.flags)
                yield PacketObservation(
                    timestamp=float(timestamp),
                    flow_key=(src, dst, src_port, dst_port, "TCP"),
                    length=len(buffer),
                    header_length=header_length + int(transport.off) * 4,
                    protocol_number=proto,
                    app_protocol=app_protocol(src_port, dst_port),
                    fin=1 if flags & dpkt.tcp.TH_FIN else 0,
                    syn=1 if flags & dpkt.tcp.TH_SYN else 0,
                    rst=1 if flags & dpkt.tcp.TH_RST else 0,
                    psh=1 if flags & dpkt.tcp.TH_PUSH else 0,
                    ack=1 if flags & dpkt.tcp.TH_ACK else 0,
                    urg=1 if flags & dpkt.tcp.TH_URG else 0,
                    is_tcp=1,
                )
                continue

            if isinstance(transport, dpkt.udp.UDP):
                src_port = int(transport.sport)
                dst_port = int(transport.dport)
                yield PacketObservation(
                    timestamp=float(timestamp),
                    flow_key=(src, dst, src_port, dst_port, "UDP"),
                    length=len(buffer),
                    header_length=header_length + 8,
                    protocol_number=proto,
                    app_protocol=app_protocol(src_port, dst_port),
                    is_udp=1,
                )
                continue

            if isinstance(transport, dpkt.icmp.ICMP):
                yield PacketObservation(
                    timestamp=float(timestamp),
                    flow_key=(src, dst, 0, 0, "ICMP"),
                    length=len(buffer),
                    header_length=header_length + 8,
                    protocol_number=proto,
                    is_icmp=1,
                )


def extract_pcap_to_records(
    pcap_path: str | Path | None,
    input_mode: str = "original_28_scaled",
    window_size: int = 10,
    scale: bool = False,
    scaler_path: str | Path | None = None,
) -> ExtractionResult:
    gaps = [
        "Experimental prototype: CICIoT2023 feature semantics are approximated from passive local packets.",
        "Labels are not inferred from pcap; scenario labels must come from controlled replay metadata.",
    ]
    if pcap_path is None:
        return ExtractionResult(records=[], gaps=gaps + ["No pcap_path supplied; interface/schema check only."], scaled=False)

    windows = completed_windows(packet_observations_from_pcap(pcap_path), window_size=window_size)
    original_rows: list[list[float | None]] = []
    unsupported_by_row: list[list[str]] = []
    for window in windows:
        row, unsupported = align_feature_row(window.extract_features())
        original_rows.append(row)
        unsupported_by_row.append(unsupported)

    scaled_rows = original_rows
    scaled = False
    if scale and original_rows:
        scaled_rows, scaled = apply_scaler(original_rows, scaler_path)

    records: list[ExtractionRecord] = []
    for index, row in enumerate(scaled_rows, start=1):
        features = selected_from_original(row) if input_mode == "selected_12_scaled" else row
        unsupported = unsupported_by_row[index - 1]
        selected_unsupported = []
        if input_mode == "selected_12_scaled":
            selected_unsupported = [name for name in unsupported if name in {"flow_duration", "Protocol Type", "Duration", "Rate", "syn_flag_number", "urg_count", "rst_count", "TCP", "UDP", "Std", "IAT", "Number"}]
        records.append(
            ExtractionRecord(
                flow_id=f"pcap-window-{index:06d}",
                input_mode=input_mode,
                features=features,
                unsupported_features=selected_unsupported if input_mode == "selected_12_scaled" else unsupported,
                approximations=[
                    "Header_Length, Duration, flag counters, Rate, and IAT are local prototype derivations.",
                    "Application protocol indicators are inferred from common ports only.",
                ],
            )
        )
    return ExtractionResult(records=records, gaps=gaps, scaled=scaled)


def write_json(path: str | Path, result: ExtractionResult) -> None:
    payload = {"scaled": result.scaled, "gaps": result.gaps, "records": result.to_payloads()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, result: ExtractionResult) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flow_id", "input_mode", "features_json", "unsupported_features_json", "approximations_json"])
        for record in result.records:
            writer.writerow(
                [
                    record.flow_id,
                    record.input_mode,
                    json.dumps(record.features),
                    json.dumps(record.unsupported_features),
                    json.dumps(record.approximations),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental passive pcap-to-final-IDS feature extractor.")
    parser.add_argument("--pcap")
    parser.add_argument("--input-mode", choices=["selected_12_scaled", "original_28_scaled"], default="original_28_scaled")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--scaler-path")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--schema", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.schema:
        print(json.dumps(schema_summary(), indent=2))
        return 0
    result = extract_pcap_to_records(args.pcap, args.input_mode, args.window_size, args.scale, args.scaler_path)
    if args.output_json:
        write_json(args.output_json, result)
    if args.output_csv:
        write_csv(args.output_csv, result)
    if not args.output_json and not args.output_csv:
        print(json.dumps({"scaled": result.scaled, "gaps": result.gaps, "records": result.to_payloads()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

