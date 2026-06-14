from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterator


Packet = dict[str, Any]
HOST_ONLY_PREFIX = "192.168.56."


def list_interfaces() -> list[str]:
    """Return local interface names without sending traffic."""

    names: set[str] = set()
    try:
        for _index, name in socket.if_nameindex():
            names.add(str(name))
    except (AttributeError, OSError):
        pass

    try:
        from scapy.all import get_if_list

        names.update(str(name) for name in get_if_list())
    except ImportError:
        pass
    return sorted(names)


def validate_interface(interface_name: str) -> dict[str, Any]:
    interfaces = list_interfaces()
    return {
        "interface": interface_name,
        "available": interface_name in interfaces,
        "known_interfaces": interfaces,
        "warning": None if interface_name in interfaces else "Interface was not found in local inventory.",
    }


def validate_host_only_ip(ip_address: str, *, field_name: str) -> None:
    if not ip_address or not ip_address.startswith(HOST_ONLY_PREFIX):
        raise ValueError(f"{field_name} must be inside the 192.168.56.0/24 host-only lab network")


def build_safe_capture_filter(node_ip: str, peer_ip: str | None = None) -> str:
    validate_host_only_ip(node_ip, field_name="node_ip")
    if peer_ip:
        validate_host_only_ip(peer_ip, field_name="peer_ip")
        return f"ip and host {node_ip} and host {peer_ip}"
    return f"ip and host {node_ip} and net 192.168.56.0/24"


def iso_from_packet_time(value: Any | None = None) -> str:
    timestamp = float(value) if value is not None else time.time()
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def tcp_flags_from_value(flags_value: int) -> dict[str, bool]:
    return {
        "fin": bool(flags_value & 0x01),
        "syn": bool(flags_value & 0x02),
        "rst": bool(flags_value & 0x04),
        "psh": bool(flags_value & 0x08),
        "ack": bool(flags_value & 0x10),
        "urg": bool(flags_value & 0x20),
    }


def packet_from_scapy(raw: Any) -> Packet | None:
    try:
        from scapy.all import ICMP, IP, TCP, UDP
    except ImportError as exc:
        raise RuntimeError("passive live observation requires scapy") from exc

    if not raw.haslayer(IP):
        return None
    ip = raw[IP]
    base: Packet = {
        "timestamp": iso_from_packet_time(getattr(raw, "time", None)),
        "src": str(ip.src),
        "dst": str(ip.dst),
        "src_ip": str(ip.src),
        "dst_ip": str(ip.dst),
        "length": int(len(bytes(raw))),
        "flags": {},
        "sport": None,
        "dport": None,
        "src_port": None,
        "dst_port": None,
    }
    if raw.haslayer(TCP):
        tcp = raw[TCP]
        src_port = int(tcp.sport)
        dst_port = int(tcp.dport)
        return {
            **base,
            "protocol": "TCP",
            "flags": tcp_flags_from_value(int(tcp.flags)),
            "sport": src_port,
            "dport": dst_port,
            "src_port": src_port,
            "dst_port": dst_port,
        }
    if raw.haslayer(UDP):
        udp = raw[UDP]
        src_port = int(udp.sport)
        dst_port = int(udp.dport)
        return {
            **base,
            "protocol": "UDP",
            "sport": src_port,
            "dport": dst_port,
            "src_port": src_port,
            "dst_port": dst_port,
        }
    if raw.haslayer(ICMP):
        return {**base, "protocol": "ICMP"}
    return {**base, "protocol": "OTHER"}


@dataclass
class PassivePacketSource:
    interface_name: str
    node_ip: str
    peer_ip: str | None = None
    max_packets: int = 30
    timeout_seconds: int = 30

    def iter_packets(self, max_packets: int | None = None) -> Iterator[Packet]:
        try:
            from scapy.all import sniff
        except ImportError as exc:
            raise RuntimeError("passive live observation requires scapy") from exc

        validation = validate_interface(self.interface_name)
        if not validation["available"]:
            raise ValueError(str(validation["warning"]))

        capture_filter = build_safe_capture_filter(self.node_ip, self.peer_ip)
        count = max_packets or self.max_packets
        captured = sniff(
            iface=self.interface_name,
            filter=capture_filter,
            count=count,
            timeout=self.timeout_seconds,
            store=True,
        )
        yielded = 0
        for raw in captured:
            packet = packet_from_scapy(raw)
            if packet is None:
                continue
            yielded += 1
            yield packet
            if yielded >= count:
                return

    def describe(self) -> dict[str, Any]:
        return {
            "source": "live",
            "mode": "real_traffic_passive_capture",
            "interface": self.interface_name,
            "node_ip": self.node_ip,
            "peer_ip": self.peer_ip,
            "capture_filter": build_safe_capture_filter(self.node_ip, self.peer_ip),
            "timeout_seconds": self.timeout_seconds,
            "passive": True,
            "stores_payload": False,
            "writes_pcap": False,
        }


def passive_packet_source(
    interface_name: str,
    max_packets: int,
    timeout_seconds: int,
    node_ip: str,
    peer_ip: str | None = None,
) -> PassivePacketSource:
    return PassivePacketSource(
        interface_name=interface_name,
        node_ip=node_ip,
        peer_ip=peer_ip,
        max_packets=max_packets,
        timeout_seconds=timeout_seconds,
    )
