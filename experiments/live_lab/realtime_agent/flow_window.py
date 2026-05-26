from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowWindow:
    flow_id: str
    max_packets: int = 10
    packets: list[dict[str, Any]] = field(default_factory=list)

    def add_packet(self, packet: dict[str, Any]) -> bool:
        self.packets.append(packet)
        return self.is_ready()

    def is_ready(self) -> bool:
        return len(self.packets) >= self.max_packets

    def summary(self) -> dict[str, object]:
        return {"flow_id": self.flow_id, "packet_count": len(self.packets), "ready": self.is_ready()}

