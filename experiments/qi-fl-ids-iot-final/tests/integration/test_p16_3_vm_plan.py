from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
SCRIPT_PATH = FINAL_ROOT / "src" / "scripts" / "16_3_validate_vm_plan.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_vm_inventory_parses_correctly() -> None:
    module = load_module("p16_3_validate_vm_plan_parse", SCRIPT_PATH)
    inventory = module.parse_vm_inventory()
    assert inventory["server"]["host_ip"] == "192.168.56.1"
    assert inventory["server"]["storage_root"] == "E:\\VirtualBox VMs\\qi-fl-ids-iot-live-lab\\"
    assert len(inventory["vms"]) == 3


def test_three_vms_exist_with_expected_ips() -> None:
    module = load_module("p16_3_validate_vm_plan_ips", SCRIPT_PATH)
    inventory = module.parse_vm_inventory()
    by_name = {vm["name"]: vm for vm in inventory["vms"]}
    assert set(by_name) == {"iot-rpi-weak", "iot-smart-watch-medium", "lab-attacker-kali"}
    assert by_name["iot-rpi-weak"]["ip"] == "192.168.56.101"
    assert by_name["iot-smart-watch-medium"]["ip"] == "192.168.56.102"
    assert by_name["lab-attacker-kali"]["ip"] == "192.168.56.103"


def test_total_ram_within_recommended_limit() -> None:
    module = load_module("p16_3_validate_vm_plan_ram", SCRIPT_PATH)
    inventory = module.parse_vm_inventory()
    total_ram = sum(int(vm["ram_mb"]) for vm in inventory["vms"])
    assert total_ram <= 4096
    assert total_ram == 4096


def test_kali_safety_doc_exists() -> None:
    assert (LIVE_LAB_ROOT / "docs" / "kali_attacker_safety_scope.md").exists()


def test_selected_scenario_doc_contains_three_scenarios() -> None:
    text = (LIVE_LAB_ROOT / "docs" / "ciciot2023_selected_attack_scenarios.md").read_text(encoding="utf-8")
    assert "icmp_flood_like" in text
    assert "tcp_syn_recon_like" in text
    assert "http_slow_like" in text


def test_validation_script_generates_ok_true() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    report = FINAL_ROOT / "outputs" / "reports" / "p16_3_vm_plan_validation.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

