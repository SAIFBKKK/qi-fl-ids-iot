from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PacketCapturePlan:
    interface: str
    enabled: bool = False
    mode: str = "dry_run"


def build_capture_plan(interface: str = "not-configured") -> PacketCapturePlan:
    return PacketCapturePlan(interface=interface, enabled=False)


def capture_packets_dry_run() -> list[bytes]:
    """P16.1 step 0 never starts packet capture."""
    return []

