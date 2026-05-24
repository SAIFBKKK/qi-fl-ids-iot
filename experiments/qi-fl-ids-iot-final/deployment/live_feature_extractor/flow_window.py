from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PacketObservation:
    timestamp: float
    flow_key: tuple[str, str, int, int, str]
    length: int
    header_length: int
    protocol_number: int
    app_protocol: str | None = None
    fin: int = 0
    syn: int = 0
    rst: int = 0
    psh: int = 0
    ack: int = 0
    urg: int = 0
    is_tcp: int = 0
    is_udp: int = 0
    is_arp: int = 0
    is_icmp: int = 0


@dataclass
class FlowWindow:
    flow_key: tuple[str, str, int, int, str]
    max_packets: int = 10
    packets: list[PacketObservation] = field(default_factory=list)

    def add(self, packet: PacketObservation) -> bool:
        if packet.flow_key != self.flow_key:
            raise ValueError("packet flow_key does not match window")
        self.packets.append(packet)
        return len(self.packets) >= self.max_packets

    def extract_features(self) -> dict[str, float]:
        if not self.packets:
            raise ValueError("cannot extract an empty flow window")
        timestamps = [packet.timestamp for packet in self.packets]
        lengths = [packet.length for packet in self.packets]
        duration = max(timestamps) - min(timestamps)
        safe_duration = duration if duration > 0 else 1e-9
        iats = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
        app_protocols = {packet.app_protocol for packet in self.packets if packet.app_protocol}
        return {
            "flow_duration": duration,
            "Header_Length": float(sum(packet.header_length for packet in self.packets)),
            "Protocol Type": float(self.packets[0].protocol_number),
            "Duration": duration,
            "Rate": float(len(self.packets)) / safe_duration,
            "fin_flag_number": float(sum(packet.fin for packet in self.packets)),
            "syn_flag_number": float(sum(packet.syn for packet in self.packets)),
            "rst_flag_number": float(sum(packet.rst for packet in self.packets)),
            "psh_flag_number": float(sum(packet.psh for packet in self.packets)),
            "ack_flag_number": float(sum(packet.ack for packet in self.packets)),
            "ack_count": float(sum(packet.ack for packet in self.packets)),
            "syn_count": float(sum(packet.syn for packet in self.packets)),
            "fin_count": float(sum(packet.fin for packet in self.packets)),
            "urg_count": float(sum(packet.urg for packet in self.packets)),
            "rst_count": float(sum(packet.rst for packet in self.packets)),
            "HTTP": 1.0 if "HTTP" in app_protocols else 0.0,
            "HTTPS": 1.0 if "HTTPS" in app_protocols else 0.0,
            "DNS": 1.0 if "DNS" in app_protocols else 0.0,
            "SSH": 1.0 if "SSH" in app_protocols else 0.0,
            "TCP": float(max(packet.is_tcp for packet in self.packets)),
            "UDP": float(max(packet.is_udp for packet in self.packets)),
            "ARP": float(max(packet.is_arp for packet in self.packets)),
            "ICMP": float(max(packet.is_icmp for packet in self.packets)),
            "Tot sum": float(sum(lengths)),
            "Min": float(min(lengths)),
            "Std": float(statistics.pstdev(lengths)) if len(lengths) > 1 else 0.0,
            "IAT": float(sum(iats) / len(iats)) if iats else 0.0,
            "Number": float(len(self.packets)),
        }


def completed_windows(packets: Iterable[PacketObservation], window_size: int) -> list[FlowWindow]:
    open_windows: dict[tuple[str, str, int, int, str], FlowWindow] = {}
    completed: list[FlowWindow] = []
    for packet in packets:
        window = open_windows.get(packet.flow_key)
        if window is None:
            window = FlowWindow(packet.flow_key, max_packets=window_size)
            open_windows[packet.flow_key] = window
        if window.add(packet):
            completed.append(window)
            del open_windows[packet.flow_key]
    completed.extend(window for window in open_windows.values() if window.packets)
    return completed


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None

