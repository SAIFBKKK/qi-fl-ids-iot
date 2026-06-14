from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class NodeConfig:
    node_id: str
    raw_dir: str
    processed_dir: str


@dataclass
class NodeProfile:
    node_id: str
    cpu_cores: int
    ram_mb: int
    device_type: str
    avg_latency_ms: float
    battery_powered: bool
    network_quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpu_cores": int(self.cpu_cores),
            "ram_mb": int(self.ram_mb),
            "device_type": self.device_type,
            "avg_latency_ms": float(self.avg_latency_ms),
            "battery_powered": bool(self.battery_powered),
            "network_quality": self.network_quality,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeProfile":
        required = (
            "node_id",
            "cpu_cores",
            "ram_mb",
            "device_type",
            "avg_latency_ms",
            "battery_powered",
            "network_quality",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing NodeProfile fields: {missing}")

        battery_powered = data["battery_powered"]
        if not isinstance(battery_powered, bool):
            raise ValueError("NodeProfile.battery_powered must be a bool")

        return cls(
            node_id=str(data["node_id"]),
            cpu_cores=int(data["cpu_cores"]),
            ram_mb=int(data["ram_mb"]),
            device_type=str(data["device_type"]),
            avg_latency_ms=float(data["avg_latency_ms"]),
            battery_powered=battery_powered,
            network_quality=str(data["network_quality"]),
        )


@dataclass
class TierAssignment:
    assigned_tier: str
    model_width: float
    local_epochs: int
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "assigned_tier": self.assigned_tier,
            "model_width": float(self.model_width),
            "local_epochs": int(self.local_epochs),
            "batch_size": int(self.batch_size),
        }
