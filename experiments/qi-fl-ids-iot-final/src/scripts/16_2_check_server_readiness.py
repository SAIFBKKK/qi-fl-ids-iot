from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DEPLOYMENT_DIR = FINAL_ROOT / "deployment"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"
COMPOSE_PATH = DEPLOYMENT_DIR / "docker-compose.final.yml"

REQUIRED_SERVICES = [
    "mosquitto",
    "final-ids-api",
    "final-mqtt-bridge",
    "online-validator",
    "live-lab-controller",
    "dashboard-p13",
    "prometheus",
    "grafana",
]

ENDPOINTS = {
    "final_ids_api_ready": "http://127.0.0.1:8014/ready",
    "final_mqtt_bridge_ready": "http://127.0.0.1:8016/ready",
    "online_validator_ready": "http://127.0.0.1:8015/ready",
    "live_lab_controller_health": "http://127.0.0.1:8020/health",
    "live_lab_controller_nodes": "http://127.0.0.1:8020/nodes",
    "dashboard_p13_health": "http://127.0.0.1:8013/health",
}

COMPOSE_CONFIG_COMMAND = [
    "docker",
    "compose",
    "-f",
    str(COMPOSE_PATH),
    "--profile",
    "online",
    "--profile",
    "live-lab",
    "config",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def service_defined(compose_text: str, service_name: str) -> bool:
    return re.search(rf"^\s{{2}}{re.escape(service_name)}:\s*$", compose_text, flags=re.MULTILINE) is not None


def check_compose_file() -> dict[str, Any]:
    compose_text = read_text(COMPOSE_PATH)
    services = {service: service_defined(compose_text, service) for service in REQUIRED_SERVICES}
    return {
        "path": str(COMPOSE_PATH),
        "exists": COMPOSE_PATH.exists(),
        "required_services": services,
        "all_required_services_defined": all(services.values()),
    }


def run_command(command: list[str], timeout: int = 45) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=DEPLOYMENT_DIR,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"ran": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.SubprocessError as exc:
        return {"ran": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "ran": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2500:],
        "stderr": completed.stderr[-2500:],
    }


def check_docker_daemon() -> dict[str, Any]:
    result = run_command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20)
    return {
        "reachable": bool(result["ran"] and result["returncode"] == 0),
        "server_version": result["stdout"].strip() if result["returncode"] == 0 else None,
        "details": result,
    }


def check_compose_config(run_compose_config: bool = True) -> dict[str, Any]:
    if not run_compose_config:
        return {
            "command": " ".join(COMPOSE_CONFIG_COMMAND),
            "ran": False,
            "returncode": None,
            "ok": "not_run",
            "stdout": "",
            "stderr": "",
        }
    result = run_command(COMPOSE_CONFIG_COMMAND, timeout=60)
    return {
        "command": " ".join(COMPOSE_CONFIG_COMMAND),
        "ran": result["ran"],
        "returncode": result["returncode"],
        "ok": bool(result["ran"] and result["returncode"] == 0),
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def probe_endpoint(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed: Any
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = body[:500]
            return {"reachable": True, "status": response.status, "body": parsed}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"reachable": True, "status": exc.code, "body": body[:500]}
    except Exception as exc:  # noqa: BLE001 - endpoint probing is optional readiness evidence.
        return {"reachable": False, "status": None, "error": str(exc)}


def check_endpoints(probe_endpoints: bool = True) -> dict[str, Any]:
    if not probe_endpoints:
        return {name: {"url": url, "reachable": "not_run", "status": None} for name, url in ENDPOINTS.items()}
    return {name: {"url": url, **probe_endpoint(url)} for name, url in ENDPOINTS.items()}


def collect_warnings(docker: dict[str, Any], compose_config: dict[str, Any], endpoints: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not docker["reachable"]:
        warnings.append("Docker Desktop daemon is not reachable; runtime checks may be unavailable.")
    if compose_config["ok"] is False:
        warnings.append("docker compose config did not complete successfully.")
    for name, result in endpoints.items():
        if result.get("reachable") is False:
            warnings.append(f"Endpoint {name} is not reachable at {result.get('url')}.")
        elif result.get("status") and int(result["status"]) >= 400:
            warnings.append(f"Endpoint {name} returned HTTP {result['status']}.")
    return warnings


def run_server_readiness(run_compose_config: bool = True, probe_endpoints: bool = True) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    compose = check_compose_file()
    docker = check_docker_daemon()
    compose_config = check_compose_config(run_compose_config=run_compose_config)
    endpoints = check_endpoints(probe_endpoints=probe_endpoints)
    warnings = collect_warnings(docker, compose_config, endpoints)
    result = {
        "generated_at": utc_now(),
        "compose": compose,
        "docker": docker,
        "compose_config": compose_config,
        "endpoints": endpoints,
        "warnings": warnings,
        "server_ports": {
            "mqtt": 1883,
            "final_ids_api": 8014,
            "final_mqtt_bridge": 8016,
            "online_validator": 8015,
            "live_lab_controller": 8020,
            "dashboard_p13": 8013,
            "prometheus": 9090,
            "grafana": 3000,
        },
    }
    result["ok"] = bool(
        compose["exists"]
        and compose["all_required_services_defined"]
        and compose_config["ok"] in {True, "not_run"}
    )
    write_reports(result)
    return result


def write_reports(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "p16_2_server_readiness.json"
    md_path = REPORT_DIR / "p16_2_server_readiness.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# P16.2 Server Readiness",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Overall status: `{'OK' if result['ok'] else 'FAILED'}`",
        f"- Compose file exists: `{result['compose']['exists']}`",
        f"- Required services defined: `{result['compose']['all_required_services_defined']}`",
        f"- Docker daemon reachable: `{result['docker']['reachable']}`",
        f"- Compose config command: `{result['compose_config']['command']}`",
        f"- Compose config status: `{result['compose_config']['ok']}`",
        "",
        "## Services",
        "",
    ]
    for service, present in result["compose"]["required_services"].items():
        lines.append(f"- `{service}`: `{present}`")

    lines.extend(["", "## Endpoints", ""])
    for name, endpoint in result["endpoints"].items():
        lines.append(
            f"- `{name}`: `{endpoint.get('url')}`, reachable=`{endpoint.get('reachable')}`, status=`{endpoint.get('status')}`"
        )

    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("- None.")

    lines.extend(["", "## VM Network", ""])
    lines.append("- Use `SERVER_IP` from the Windows VirtualBox Host-Only adapter.")
    lines.append("- Future VM root: `E:\\VirtualBox VMs\\qi-fl-ids-iot-live-lab\\`.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P16.2 server readiness checker.")
    parser.add_argument("--skip-compose-config", action="store_true")
    parser.add_argument("--skip-endpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_server_readiness(
        run_compose_config=not args.skip_compose_config,
        probe_endpoints=not args.skip_endpoints,
    )
    print(json.dumps({"ok": result["ok"], "warnings": result["warnings"], "report_dir": str(REPORT_DIR)}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

