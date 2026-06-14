from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPECTED_KALI_IP = "192.168.56.103"
LAB_HOSTS = {
    "server": "192.168.56.1",
    "iot-drone-sitl": "192.168.56.101",
    "iot-smart-watch-medium": "192.168.56.102",
}
SERVER_ENDPOINTS = {
    "live_lab_controller_health": "http://192.168.56.1:8020/health",
    "final_ids_api_ready": "http://192.168.56.1:8014/ready",
    "final_mqtt_bridge_ready": "http://192.168.56.1:8016/ready",
    "online_validator_ready": "http://192.168.56.1:8015/ready",
    "dashboard_health": "http://192.168.56.1:8013/health",
}
INVENTORY_TOOLS = ["hping3", "nmap", "python3", "curl", "tcpdump", "tshark"]

REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"
JSON_REPORT = REPORT_DIR / "p16_10_kali_readiness.json"
MD_REPORT = REPORT_DIR / "p16_10_kali_readiness.md"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_probably_kali() -> bool:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return False
    text = os_release.read_text(encoding="utf-8", errors="ignore").lower()
    return "kali" in text


def get_local_ips() -> list[str]:
    ips: set[str] = set()
    try:
        hostname = socket.gethostname()
        for value in socket.gethostbyname_ex(hostname)[2]:
            if value and not value.startswith("127."):
                ips.add(value)
    except OSError:
        pass

    if platform.system().lower() != "windows" and shutil.which("ip"):
        try:
            result = subprocess.run(
                ["ip", "-o", "-4", "addr", "show"],
                check=False,
                text=True,
                capture_output=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if "inet" in parts:
                    address = parts[parts.index("inet") + 1].split("/", 1)[0]
                    if address and not address.startswith("127."):
                        ips.add(address)
        except (OSError, subprocess.SubprocessError):
            pass

    return sorted(ips)


def ping_fixed_host(ip_address: str) -> dict[str, Any]:
    system = platform.system().lower()
    if system == "windows":
        command = ["ping", "-n", "1", "-w", "1000", ip_address]
    else:
        command = ["ping", "-c", "1", "-W", "1", ip_address]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=3,
        )
        return {
            "reachable": result.returncode == 0,
            "method": "single_fixed_host_ping",
            "target": ip_address,
            "returncode": result.returncode,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "reachable": False,
            "method": "single_fixed_host_ping",
            "target": ip_address,
            "warning": str(exc),
        }


def check_endpoint(name: str, url: str) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "p16-10-kali-readiness/1.0"})
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read(512).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": url,
                "body_preview": body[:160],
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "url": url, "warning": str(exc)}


def inventory_tools() -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for tool in INVENTORY_TOOLS:
        path = shutil.which(tool)
        inventory[tool] = {
            "present": bool(path),
            "path": path,
            "inventory_only": True,
            "executed": False,
        }
    return inventory


def build_readiness() -> dict[str, Any]:
    hostname = socket.gethostname()
    local_ips = get_local_ips()
    warnings: list[str] = []
    if platform.system().lower() == "windows":
        warnings.append("This readiness check appears to be running on Windows, not inside the Kali VM.")
    if not is_probably_kali():
        warnings.append("Kali OS marker was not detected; run this script inside lab-attacker-kali for final readiness evidence.")
    if EXPECTED_KALI_IP not in local_ips:
        warnings.append(f"Expected host-only IP {EXPECTED_KALI_IP} was not found in local interface inventory.")

    host_reachability = {name: ping_fixed_host(ip_address) for name, ip_address in LAB_HOSTS.items()}
    endpoints = {name: check_endpoint(name, url) for name, url in SERVER_ENDPOINTS.items()}
    tools = inventory_tools()

    server_ready = all(item.get("ok") for item in endpoints.values())
    lab_hosts_seen = host_reachability["server"].get("reachable", False)
    expected_ip_seen = EXPECTED_KALI_IP in local_ips

    return {
        "generated_at": utc_now(),
        "hostname": hostname,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "is_probably_kali": is_probably_kali(),
        },
        "expected_kali_ip": EXPECTED_KALI_IP,
        "local_ips": local_ips,
        "expected_ip_seen": expected_ip_seen,
        "lab_hosts": LAB_HOSTS,
        "host_reachability": host_reachability,
        "server_endpoints": endpoints,
        "tool_inventory": tools,
        "safety": {
            "inventory_only": True,
            "no_scenarios_executed": True,
            "no_subnet_scan": True,
            "no_live_capture": True,
            "no_traffic_generation_tools_executed": True,
        },
        "warnings": warnings,
        "ready_for_future_controlled_design": bool(expected_ip_seen and server_ready and lab_hosts_seen),
    }


def write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# P16.10 Kali Readiness Runtime Check",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Hostname: `{result['hostname']}`",
        f"- Platform: `{result['platform']['system']} {result['platform']['release']}`",
        f"- Kali marker detected: `{result['platform']['is_probably_kali']}`",
        f"- Expected IP: `{result['expected_kali_ip']}`",
        f"- Expected IP seen: `{result['expected_ip_seen']}`",
        f"- Ready for future controlled design: `{result['ready_for_future_controlled_design']}`",
        "",
        "## Local IPs",
        "",
    ]
    lines.extend(f"- `{ip}`" for ip in result["local_ips"])
    lines.extend(["", "## Fixed Lab Host Reachability", ""])
    for name, item in result["host_reachability"].items():
        lines.append(f"- `{name}` `{item['target']}`: `{item.get('reachable')}`")
    lines.extend(["", "## Server Endpoints", ""])
    for name, item in result["server_endpoints"].items():
        lines.append(f"- `{name}`: `{item.get('ok')}` `{item.get('url')}`")
    lines.extend(["", "## Tool Inventory Only", ""])
    for name, item in result["tool_inventory"].items():
        lines.append(f"- `{name}` present: `{item.get('present')}` executed: `{item.get('executed')}`")
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This check does not execute scenario tools, does not scan a subnet, does not start live capture, and does not generate traffic.",
        ]
    )
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = build_readiness()
    write_reports(result)
    print(json.dumps({"ok": True, "ready_for_future_controlled_design": result["ready_for_future_controlled_design"], "warnings": result["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

