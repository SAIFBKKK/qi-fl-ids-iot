from __future__ import annotations

import ctypes
import os
import platform
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    node_id: str
    hostname: str
    cpu_count: int
    ram_gb: float
    device_type: str
    mqtt_topic: str

    def as_registration_payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb,
            "device_type": self.device_type,
            "mqtt_topic": self.mqtt_topic,
        }


def detect_ram_gb() -> float:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages and page_size:
                return round((pages * page_size) / (1024**3), 2)
        except (OSError, ValueError):
            pass

    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 2)

    return 0.0


def collect_hardware_profile(node_id: str, device_type: str = "iot_vm") -> HardwareProfile:
    return HardwareProfile(
        node_id=node_id,
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=detect_ram_gb(),
        device_type=device_type,
        mqtt_topic=f"ids/flows/{node_id}",
    )

