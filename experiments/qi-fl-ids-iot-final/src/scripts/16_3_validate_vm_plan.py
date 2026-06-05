from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"
INVENTORY_PATH = LIVE_LAB_ROOT / "configs" / "vm_inventory.yaml"

EXPECTED_SERVER_IP = "192.168.56.1"
EXPECTED_VM_IPS = {
    "iot-drone-sitl": "192.168.56.101",
    "iot-smart-watch-medium": "192.168.56.102",
    "lab-attacker-kali": "192.168.56.103",
}
EXPECTED_VM_NAMES = set(EXPECTED_VM_IPS)
RECOMMENDED_MAX_RAM_MB = 4096

REQUIRED_DOCS = [
    FINAL_ROOT / "outputs" / "reports" / "p16_3_virtualbox_vm_plan.md",
    LIVE_LAB_ROOT / "docs" / "vm_resource_plan.md",
    LIVE_LAB_ROOT / "docs" / "vm_network_static_ips.md",
    LIVE_LAB_ROOT / "docs" / "kali_attacker_safety_scope.md",
    LIVE_LAB_ROOT / "docs" / "ciciot2023_selected_attack_scenarios.md",
    LIVE_LAB_ROOT / "docs" / "vm_creation_checklist.md",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def parse_scalar(raw: str) -> Any:
    value = raw.strip().strip('"').strip("'")
    value = value.replace("\\\\", "\\")
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def parse_vm_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    inventory_path = Path(path)
    data: dict[str, Any] = {"server": {}, "vms": []}
    section: str | None = None
    current_vm: dict[str, Any] | None = None

    for raw_line in inventory_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "server:":
            section = "server"
            current_vm = None
            continue
        if stripped == "vms:":
            section = "vms"
            current_vm = None
            continue
        if section == "server" and line.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            data["server"][key.strip()] = parse_scalar(value)
            continue
        if section == "vms" and stripped.startswith("- "):
            item = stripped[2:]
            current_vm = {}
            data["vms"].append(current_vm)
            if ":" in item:
                key, value = item.split(":", 1)
                current_vm[key.strip()] = parse_scalar(value)
            continue
        if section == "vms" and current_vm is not None and line.startswith("    ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_vm[key.strip()] = parse_scalar(value)
            continue

    return data


def check_required_docs() -> dict[str, Any]:
    missing = [rel(path) for path in REQUIRED_DOCS if not path.exists()]
    return {"ok": not missing, "missing": missing}


def check_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    server = inventory.get("server", {})
    vms = inventory.get("vms", [])
    names = {str(vm.get("name")) for vm in vms}
    ip_by_name = {str(vm.get("name")): str(vm.get("ip")) for vm in vms}
    total_ram_mb = sum(int(vm.get("ram_mb", 0)) for vm in vms)
    return {
        "inventory_exists": INVENTORY_PATH.exists(),
        "server_ip": server.get("host_ip"),
        "server_ip_ok": server.get("host_ip") == EXPECTED_SERVER_IP,
        "vm_names": sorted(names),
        "all_vms_defined": names == EXPECTED_VM_NAMES,
        "vm_ips": ip_by_name,
        "vm_ips_ok": ip_by_name == EXPECTED_VM_IPS,
        "total_ram_mb": total_ram_mb,
        "ram_within_recommended_limit": total_ram_mb <= RECOMMENDED_MAX_RAM_MB,
    }


def text_files(root: Path) -> list[Path]:
    suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh", ".example"}
    return [path for path in root.rglob("*") if path.is_file() and (path.suffix.lower() in suffixes or path.name.endswith(".env.example"))]


def concrete_offensive_command_patterns() -> list[re.Pattern[str]]:
    terms = [
        "h" + "ping3",
        "n" + "map",
        "slow" + "loris",
        "hy" + "dra",
        "ett" + "ercap",
        "mir" + "ai",
        "golang-http" + "flood",
    ]
    command = "|".join(re.escape(term) for term in terms)
    return [
        re.compile(rf"^\s*(?:\$|>|PS>|sudo\s+)?(?:sudo\s+)?(?:{command})\b", re.IGNORECASE),
        re.compile(rf"\b(?:python|python3|bash|sh)\s+\S*(?:{command})\S*", re.IGNORECASE),
    ]


def check_no_concrete_offensive_commands() -> dict[str, Any]:
    patterns = concrete_offensive_command_patterns()
    violations: list[dict[str, Any]] = []
    for path in text_files(LIVE_LAB_ROOT):
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    violations.append({"file": rel(path), "line": line_number, "text": line.strip()})
    return {
        "ok": not violations,
        "violations": violations,
        "note": "Tool names are allowed as documentary references, but executable command lines are not allowed.",
    }


def check_scenario_doc() -> dict[str, Any]:
    path = LIVE_LAB_ROOT / "docs" / "ciciot2023_selected_attack_scenarios.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    scenarios = ["icmp_flood_like", "tcp_syn_recon_like", "http_slow_like"]
    present = {scenario: scenario in text for scenario in scenarios}
    return {"ok": all(present.values()), "present": present}


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = parse_vm_inventory(INVENTORY_PATH) if INVENTORY_PATH.exists() else {"server": {}, "vms": []}
    docs = check_required_docs()
    inventory_check = check_inventory(inventory)
    safety = check_no_concrete_offensive_commands()
    scenarios = check_scenario_doc()
    result = {
        "generated_at": utc_now(),
        "docs": docs,
        "inventory": inventory_check,
        "safety": safety,
        "scenarios": scenarios,
    }
    result["ok"] = all(
        [
            docs["ok"],
            inventory_check["inventory_exists"],
            inventory_check["server_ip_ok"],
            inventory_check["all_vms_defined"],
            inventory_check["vm_ips_ok"],
            inventory_check["ram_within_recommended_limit"],
            safety["ok"],
            scenarios["ok"],
        ]
    )
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_3_vm_plan_validation.json"
    md_path = REPORT_DIR / "p16_3_vm_plan_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# P16.3 VM Plan Validation",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Required docs present: `{result['docs']['ok']}`",
        f"- Inventory exists: `{result['inventory']['inventory_exists']}`",
        f"- Server IP: `{result['inventory']['server_ip']}`",
        f"- Server IP OK: `{result['inventory']['server_ip_ok']}`",
        f"- VM names: `{', '.join(result['inventory']['vm_names'])}`",
        f"- VM IPs OK: `{result['inventory']['vm_ips_ok']}`",
        f"- Total VM RAM MB: `{result['inventory']['total_ram_mb']}`",
        f"- RAM within recommended limit: `{result['inventory']['ram_within_recommended_limit']}`",
        f"- Concrete offensive command scan: `{'OK' if result['safety']['ok'] else 'FAILED'}`",
        f"- Scenario docs complete: `{result['scenarios']['ok']}`",
        "",
    ]
    if result["docs"]["missing"]:
        lines.extend(["## Missing Docs", ""])
        lines.extend(f"- `{item}`" for item in result["docs"]["missing"])
    if result["safety"]["violations"]:
        lines.extend(["## Safety Violations", ""])
        lines.extend(f"- `{item['file']}` line `{item['line']}`" for item in result["safety"]["violations"])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_validation()
    print(json.dumps({"ok": result["ok"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

