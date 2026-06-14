from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class HardwareProfile:
    hostname: str
    cpu_count: int
    ram_gb: float
    platform: str


def _ram_gb() -> float:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 2)
        except (OSError, ValueError):
            return 0.0
    return 0.0


def get_hardware_profile() -> dict[str, object]:
    profile = HardwareProfile(
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        platform=platform.platform(),
    )
    return asdict(profile)

