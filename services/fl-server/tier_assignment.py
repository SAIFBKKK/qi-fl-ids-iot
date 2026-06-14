from __future__ import annotations

from dataclasses import dataclass


VALID_TIERS = ("weak", "medium", "powerful")


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    cpu_cores: int
    ram_mb: int
    device_type: str = "docker_node"
    network_quality: str = "medium"
    battery_powered: bool = False
    tier_override: str | None = None


def normalize_tier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in VALID_TIERS:
        raise ValueError(f"Invalid tier override {value!r}; expected one of {VALID_TIERS}")
    return normalized


def assign_tier(profile: NodeProfile) -> str:
    override = normalize_tier(profile.tier_override)
    if override is not None:
        return override

    if profile.cpu_cores <= 2 or profile.ram_mb < 2048:
        return "weak"
    if profile.cpu_cores <= 4 or profile.ram_mb < 4096:
        return "medium"
    return "powerful"
