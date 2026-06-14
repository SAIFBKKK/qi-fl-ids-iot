from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from tier_assignment import NodeProfile, assign_tier


@dataclass
class RegisteredNode:
    node_id: str
    cpu_cores: int
    ram_mb: int
    device_type: str
    network_quality: str
    battery_powered: bool
    assigned_tier: str
    model_version: str
    model_source: str
    status: str = "registered"
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_heartbeat: str | None = None


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, RegisteredNode] = {}
        self._lock = Lock()

    def register(self, payload: dict[str, Any]) -> RegisteredNode:
        profile = NodeProfile(
            node_id=str(payload["node_id"]),
            cpu_cores=int(payload.get("cpu_cores", 1)),
            ram_mb=int(payload.get("ram_mb", 1024)),
            device_type=str(payload.get("device_type", "docker_node")),
            network_quality=str(payload.get("network_quality", "medium")),
            battery_powered=bool(payload.get("battery_powered", False)),
            tier_override=payload.get("tier_override"),
        )
        tier = assign_tier(profile)
        now = datetime.now(UTC).isoformat()

        with self._lock:
            existing = self._nodes.get(profile.node_id)
            registered_at = existing.registered_at if existing is not None else now
            node = RegisteredNode(
                node_id=profile.node_id,
                cpu_cores=profile.cpu_cores,
                ram_mb=profile.ram_mb,
                device_type=profile.device_type,
                network_quality=profile.network_quality,
                battery_powered=profile.battery_powered,
                assigned_tier=tier,
                model_version="placeholder",
                model_source="local_registry",
                status="connected",
                registered_at=registered_at,
                updated_at=now,
                last_heartbeat=now,
            )
            self._nodes[profile.node_id] = node
            return node

    def heartbeat(self, node_id: str) -> dict[str, Any] | None:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.last_heartbeat = now
            node.updated_at = now
            node.status = "connected"
            return asdict(node)

    def refresh_heartbeat_statuses(self, stale_seconds: int, timeout_seconds: int) -> None:
        now = datetime.now(UTC)
        with self._lock:
            for node in self._nodes.values():
                if not node.last_heartbeat:
                    continue
                try:
                    last_heartbeat = datetime.fromisoformat(
                        node.last_heartbeat.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                age = (now - last_heartbeat).total_seconds()
                if age > timeout_seconds:
                    node.status = "disconnected"
                elif age > stale_seconds:
                    node.status = "stale"

    def list_nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(node) for node in sorted(self._nodes.values(), key=lambda item: item.node_id)]

    def counts_by_tier(self) -> dict[str, int]:
        counts = {"weak": 0, "medium": 0, "powerful": 0}
        with self._lock:
            for node in self._nodes.values():
                counts[node.assigned_tier] = counts.get(node.assigned_tier, 0) + 1
        return counts

    def total(self) -> int:
        with self._lock:
            return len(self._nodes)
