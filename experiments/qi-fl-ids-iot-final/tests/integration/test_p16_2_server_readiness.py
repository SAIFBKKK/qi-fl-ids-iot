from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
LIVE_LAB_ROOT = REPO_ROOT / "experiments" / "live_lab"
SCRIPT_PATH = FINAL_ROOT / "src" / "scripts" / "16_2_check_server_readiness.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_server_readiness_script_importable() -> None:
    module = load_module("p16_2_server_readiness", SCRIPT_PATH)
    assert module.COMPOSE_PATH.exists()
    assert "final-ids-api" in module.REQUIRED_SERVICES


def test_p16_2_reports_and_docs_exist() -> None:
    assert (FINAL_ROOT / "outputs" / "reports" / "p16_2_server_readiness_plan.md").exists()
    assert (FINAL_ROOT / "outputs" / "reports" / "p16_2_vm_connection_card.md").exists()
    assert (LIVE_LAB_ROOT / "docs" / "server_network_readiness.md").exists()


def test_compose_contains_required_server_services() -> None:
    text = (FINAL_ROOT / "deployment" / "docker-compose.final.yml").read_text(encoding="utf-8")
    for service in ["live-lab-controller", "final-mqtt-bridge", "final-ids-api", "online-validator"]:
        assert f"{service}:" in text


def test_server_readiness_generates_json_and_md_without_runtime_services() -> None:
    module = load_module("p16_2_server_readiness_generate", SCRIPT_PATH)
    result = module.run_server_readiness(run_compose_config=False, probe_endpoints=False)
    assert result["ok"] is True
    assert result["compose"]["all_required_services_defined"] is True
    json_path = FINAL_ROOT / "outputs" / "reports" / "p16_2_server_readiness.json"
    md_path = FINAL_ROOT / "outputs" / "reports" / "p16_2_server_readiness.md"
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["ok"] is True

