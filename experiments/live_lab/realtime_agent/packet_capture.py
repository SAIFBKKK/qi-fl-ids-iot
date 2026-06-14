from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


Packet = dict[str, Any]


def iso_timestamp(offset_ms: int = 0) -> str:
    value = datetime.now(UTC) + timedelta(milliseconds=offset_ms)
    return value.isoformat().replace("+00:00", "Z")


def normalize_packet(
    *,
    timestamp: str,
    protocol: str,
    length: int,
    src: str,
    dst: str,
    flags: dict[str, bool] | None = None,
    sport: int | None = None,
    dport: int | None = None,
) -> Packet:
    return {
        "timestamp": timestamp,
        "protocol": protocol.upper(),
        "length": int(length),
        "src": src,
        "dst": dst,
        "flags": flags or {},
        "sport": sport,
        "dport": dport,
    }


@dataclass
class SyntheticPacketSource:
    """Deterministic packet source used by default for safe dry-runs."""

    packet_count: int = 30

    def iter_packets(self, max_packets: int | None = None) -> Iterator[Packet]:
        count = max_packets or self.packet_count
        for index in range(count):
            protocol = ("TCP", "UDP", "ICMP")[index % 3]
            flags = {
                "fin": False,
                "syn": protocol == "TCP" and index % 10 == 0,
                "rst": False,
                "psh": protocol == "TCP" and index % 5 == 0,
                "ack": protocol == "TCP" and index % 2 == 0,
                "urg": False,
            }
            sport = 12000 + index if protocol in {"TCP", "UDP"} else None
            dport = [80, 443, 53, 22, 1883][index % 5] if protocol in {"TCP", "UDP"} else None
            yield normalize_packet(
                timestamp=iso_timestamp(index * 10),
                protocol=protocol,
                length=64 + (index % 9) * 8,
                src=f"192.168.56.{101 + (index % 2)}",
                dst="192.168.56.1",
                flags=flags,
                sport=sport,
                dport=dport,
            )

    def describe(self) -> dict[str, Any]:
        return {"source": "synthetic", "packet_count": self.packet_count, "safe_default": True}


@dataclass
class OptionalPcapPacketSource:
    pcap_path: str | Path

    def iter_packets(self, max_packets: int | None = None) -> Iterator[Packet]:
        path = Path(self.pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"pcap file not found: {path}")
        yielded = 0
        for packet in self._iter_with_scapy(path):
            yield packet
            yielded += 1
            if max_packets and yielded >= max_packets:
                return
        if yielded:
            return
        for packet in self._iter_with_dpkt(path):
            yield packet
            yielded += 1
            if max_packets and yielded >= max_packets:
                return
        if yielded == 0:
            raise RuntimeError("pcap reading requires scapy or dpkt for this prototype")

    def _iter_with_scapy(self, path: Path) -> Iterator[Packet]:
        try:
            from scapy.all import ARP, ICMP, IP, TCP, UDP, PcapReader
        except ImportError:
            return

        with PcapReader(str(path)) as reader:
            for raw in reader:
                timestamp = datetime.fromtimestamp(float(raw.time), UTC).isoformat().replace("+00:00", "Z")
                if raw.haslayer(IP):
                    ip = raw[IP]
                    src = str(ip.src)
                    dst = str(ip.dst)
                elif raw.haslayer(ARP):
                    arp = raw[ARP]
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="ARP",
                        length=len(bytes(raw)),
                        src=str(getattr(arp, "psrc", "")),
                        dst=str(getattr(arp, "pdst", "")),
                    )
                    continue
                else:
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="OTHER",
                        length=len(bytes(raw)),
                        src="unknown",
                        dst="unknown",
                    )
                    continue

                if raw.haslayer(TCP):
                    tcp = raw[TCP]
                    flags_value = int(tcp.flags)
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="TCP",
                        length=len(bytes(raw)),
                        src=src,
                        dst=dst,
                        flags={
                            "fin": bool(flags_value & 0x01),
                            "syn": bool(flags_value & 0x02),
                            "rst": bool(flags_value & 0x04),
                            "psh": bool(flags_value & 0x08),
                            "ack": bool(flags_value & 0x10),
                            "urg": bool(flags_value & 0x20),
                        },
                        sport=int(tcp.sport),
                        dport=int(tcp.dport),
                    )
                elif raw.haslayer(UDP):
                    udp = raw[UDP]
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="UDP",
                        length=len(bytes(raw)),
                        src=src,
                        dst=dst,
                        sport=int(udp.sport),
                        dport=int(udp.dport),
                    )
                elif raw.haslayer(ICMP):
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="ICMP",
                        length=len(bytes(raw)),
                        src=src,
                        dst=dst,
                    )
                else:
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="OTHER",
                        length=len(bytes(raw)),
                        src=src,
                        dst=dst,
                    )

    def _iter_with_dpkt(self, path: Path) -> Iterator[Packet]:
        try:
            import socket

            import dpkt
        except ImportError:
            return

        with path.open("rb") as handle:
            reader = dpkt.pcap.Reader(handle)
            for timestamp_value, buffer in reader:
                try:
                    eth = dpkt.ethernet.Ethernet(buffer)
                except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                    continue
                timestamp = datetime.fromtimestamp(float(timestamp_value), UTC).isoformat().replace("+00:00", "Z")
                if not isinstance(eth.data, dpkt.ip.IP):
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="OTHER",
                        length=len(buffer),
                        src="unknown",
                        dst="unknown",
                    )
                    continue
                ip = eth.data
                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    flags_value = int(tcp.flags)
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="TCP",
                        length=len(buffer),
                        src=src,
                        dst=dst,
                        flags={
                            "fin": bool(flags_value & dpkt.tcp.TH_FIN),
                            "syn": bool(flags_value & dpkt.tcp.TH_SYN),
                            "rst": bool(flags_value & dpkt.tcp.TH_RST),
                            "psh": bool(flags_value & dpkt.tcp.TH_PUSH),
                            "ack": bool(flags_value & dpkt.tcp.TH_ACK),
                            "urg": bool(flags_value & dpkt.tcp.TH_URG),
                        },
                        sport=int(tcp.sport),
                        dport=int(tcp.dport),
                    )
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="UDP",
                        length=len(buffer),
                        src=src,
                        dst=dst,
                        sport=int(udp.sport),
                        dport=int(udp.dport),
                    )
                elif isinstance(ip.data, dpkt.icmp.ICMP):
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="ICMP",
                        length=len(buffer),
                        src=src,
                        dst=dst,
                    )
                else:
                    yield normalize_packet(
                        timestamp=timestamp,
                        protocol="OTHER",
                        length=len(buffer),
                        src=src,
                        dst=dst,
                    )

    def describe(self) -> dict[str, Any]:
        return {"source": "pcap", "pcap_path": str(self.pcap_path), "passive": True}


@dataclass
class OptionalLivePacketSource:
    interface: str | None = None
    allow_live_capture: bool = False
    timeout_seconds: int = 10

    def iter_packets(self, max_packets: int | None = None) -> Iterator[Packet]:
        if not self.allow_live_capture:
            raise PermissionError("live packet observation requires --allow-live-capture")
        try:
            from scapy.all import ARP, ICMP, IP, TCP, UDP, sniff
        except ImportError as exc:
            raise RuntimeError("live packet observation requires scapy") from exc

        captured = sniff(
            iface=self.interface,
            count=max_packets or 30,
            timeout=self.timeout_seconds,
            store=True,
        )
        for raw in captured:
            timestamp = datetime.fromtimestamp(float(raw.time), UTC).isoformat().replace("+00:00", "Z")
            if raw.haslayer(IP):
                ip = raw[IP]
                src = str(ip.src)
                dst = str(ip.dst)
            elif raw.haslayer(ARP):
                arp = raw[ARP]
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="ARP",
                    length=len(bytes(raw)),
                    src=str(getattr(arp, "psrc", "")),
                    dst=str(getattr(arp, "pdst", "")),
                )
                continue
            else:
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="OTHER",
                    length=len(bytes(raw)),
                    src="unknown",
                    dst="unknown",
                )
                continue

            if raw.haslayer(TCP):
                tcp = raw[TCP]
                flags_value = int(tcp.flags)
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="TCP",
                    length=len(bytes(raw)),
                    src=src,
                    dst=dst,
                    flags={
                        "fin": bool(flags_value & 0x01),
                        "syn": bool(flags_value & 0x02),
                        "rst": bool(flags_value & 0x04),
                        "psh": bool(flags_value & 0x08),
                        "ack": bool(flags_value & 0x10),
                        "urg": bool(flags_value & 0x20),
                    },
                    sport=int(tcp.sport),
                    dport=int(tcp.dport),
                )
            elif raw.haslayer(UDP):
                udp = raw[UDP]
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="UDP",
                    length=len(bytes(raw)),
                    src=src,
                    dst=dst,
                    sport=int(udp.sport),
                    dport=int(udp.dport),
                )
            elif raw.haslayer(ICMP):
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="ICMP",
                    length=len(bytes(raw)),
                    src=src,
                    dst=dst,
                )
            else:
                yield normalize_packet(
                    timestamp=timestamp,
                    protocol="OTHER",
                    length=len(bytes(raw)),
                    src=src,
                    dst=dst,
                )

    def describe(self) -> dict[str, Any]:
        return {
            "source": "live",
            "interface": self.interface,
            "allow_live_capture": self.allow_live_capture,
            "passive": True,
        }


def choose_packet_source(
    *,
    pcap: str | None = None,
    interface: str | None = None,
    allow_live_capture: bool = False,
    synthetic_packet_count: int = 30,
) -> SyntheticPacketSource | OptionalPcapPacketSource | OptionalLivePacketSource:
    if pcap:
        return OptionalPcapPacketSource(pcap)
    if allow_live_capture:
        return OptionalLivePacketSource(interface=interface, allow_live_capture=True)
    return SyntheticPacketSource(packet_count=synthetic_packet_count)
