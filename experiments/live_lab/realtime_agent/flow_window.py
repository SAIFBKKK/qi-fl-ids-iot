from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4


Packet = dict[str, Any]


@dataclass
class PacketWindow:
    window_size: int = 30
    flow_id: str = field(default_factory=lambda: f"p16-7-window-{uuid4().hex[:12]}")
    packets: list[Packet] = field(default_factory=list)

    def add_packet(self, packet: Packet) -> bool:
        self.packets.append(packet)
        return self.is_ready()

    def is_ready(self) -> bool:
        return len(self.packets) >= self.window_size

    def flush(self) -> list[Packet]:
        packets = list(self.packets)
        self.packets.clear()
        return packets

    def summary(self) -> dict[str, Any]:
        protocols: dict[str, int] = {}
        total_length = 0
        for packet in self.packets:
            protocol = str(packet.get("protocol", "OTHER")).upper()
            protocols[protocol] = protocols.get(protocol, 0) + 1
            total_length += int(packet.get("length", 0) or 0)
        return {
            "flow_id": self.flow_id,
            "window_size": self.window_size,
            "packet_count": len(self.packets),
            "ready": self.is_ready(),
            "protocol_counts": protocols,
            "total_length": total_length,
        }


def packet_window_from_packets(packets: Iterable[Packet], window_size: int = 30, flow_id: str | None = None) -> PacketWindow:
    window = PacketWindow(window_size=window_size, flow_id=flow_id or f"p16-7-window-{uuid4().hex[:12]}")
    for packet in packets:
        window.add_packet(packet)
    return window


def group_packets_into_windows(packets: Iterable[Packet], window_size: int = 30) -> list[PacketWindow]:
    windows: list[PacketWindow] = []
    current = PacketWindow(window_size=window_size)
    for packet in packets:
        if current.add_packet(packet):
            windows.append(packet_window_from_packets(current.flush(), window_size=window_size, flow_id=current.flow_id))
            current = PacketWindow(window_size=window_size)
    return windows
