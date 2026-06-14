from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock

from tier_assignment import assign_tier


MODEL_ID = "p8_fedavg_qga_l1"
SELECTED_MASK_ID = "conservative_seed_42"
SUPPORTED_INPUT_MODES = ("selected_12_scaled", "original_28_scaled")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class NodeRegistration:
    node_id: str
    hostname: str
    cpu_count: int
    ram_gb: float
    device_type: str
    mqtt_topic: str | None = None


@dataclass
class RegisteredNode:
    node_id: str
    hostname: str
    cpu_count: int
    ram_gb: float
    device_type: str
    requested_mqtt_topic: str | None
    assigned_tier: str
    registered_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def assignment(self) -> dict[str, object]:
        return build_assignment(self.node_id, self.assigned_tier)


def build_assignment(node_id: str, assigned_tier: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "assigned_tier": assigned_tier,
        "model_id": MODEL_ID,
        "selected_mask_id": SELECTED_MASK_ID,
        "supported_input_modes": list(SUPPORTED_INPUT_MODES),
        "mqtt_publish_topic": f"ids/flows/{node_id}",
        "mqtt_prediction_topic": f"ids/predictions/{node_id}",
        "mqtt_alert_topic": f"ids/alerts/{node_id}",
    }


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, RegisteredNode] = {}
        self._lock = Lock()

    def register(self, registration: NodeRegistration) -> RegisteredNode:
        tier = assign_tier(registration.cpu_count, registration.ram_gb, registration.device_type)
        now = utc_now()
        with self._lock:
            existing = self._nodes.get(registration.node_id)
            registered_at = existing.registered_at if existing else now
            node = RegisteredNode(
                node_id=registration.node_id,
                hostname=registration.hostname,
                cpu_count=int(registration.cpu_count),
                ram_gb=float(registration.ram_gb),
                device_type=registration.device_type,
                requested_mqtt_topic=registration.mqtt_topic,
                assigned_tier=tier,
                registered_at=registered_at,
                updated_at=now,
            )
            self._nodes[registration.node_id] = node
            return node

    def list_nodes(self) -> list[dict[str, object]]:
        with self._lock:
            return [node.as_dict() for node in sorted(self._nodes.values(), key=lambda item: item.node_id)]

    def assignments(self) -> list[dict[str, object]]:
        with self._lock:
            return [node.assignment() for node in sorted(self._nodes.values(), key=lambda item: item.node_id)]

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)
