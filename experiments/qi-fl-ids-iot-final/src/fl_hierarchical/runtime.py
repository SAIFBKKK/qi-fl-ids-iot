"""Runtime helpers for P6 hierarchical Flower executions."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fl_l1.scenario_loader import repo_path


@dataclass(frozen=True)
class HierarchicalRunPaths:
    run_id: str
    task: str
    scenario_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    latest_run_path: Path


def make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def parse_address(address: str) -> tuple[str, int]:
    if ":" not in address:
        raise ValueError(f"Flower address must be host:port, got {address!r}")
    host, raw_port = address.rsplit(":", 1)
    return host or "127.0.0.1", int(raw_port)


def is_port_available(address: str) -> bool:
    host, port = parse_address(address)
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def assert_port_available(address: str) -> None:
    if not is_port_available(address):
        _, port = parse_address(address)
        raise RuntimeError(
            f"Flower server port is already in use: {address}. "
            f"On Windows, inspect it with: netstat -ano | findstr :{port}"
        )


def configured_address(config: dict[str, Any], override: str | None = None) -> str:
    return str(override or config.get("flower", {}).get("address", "127.0.0.1:8081"))


def scenario_run_dir(
    config: dict[str, Any],
    repo_root: Path,
    *,
    task: str,
    alpha: float,
    clients: int,
) -> Path:
    return repo_path(repo_root, config["outputs"]["run_dir"]) / task / alpha_dir(alpha) / f"k{clients}"


def prepare_run_paths(
    *,
    config: dict[str, Any],
    repo_root: Path,
    task: str,
    alpha: float,
    clients: int,
    run_id: str | None = None,
    mark_latest: bool = True,
) -> HierarchicalRunPaths:
    resolved_run_id = run_id or make_run_id()
    scenario_dir = scenario_run_dir(config, repo_root, task=task, alpha=alpha, clients=clients)
    run_dir = scenario_dir / "runs" / resolved_run_id
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    for directory in [checkpoints_dir, artifacts_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    latest_run_path = scenario_dir / "latest_run.json"
    if mark_latest:
        latest_run_path.parent.mkdir(parents=True, exist_ok=True)
        latest_run_path.write_text(
            json.dumps(
                {
                    "run_id": resolved_run_id,
                    "task": task,
                    "run_dir": run_dir.relative_to(repo_root).as_posix(),
                    "logs_dir": logs_dir.relative_to(repo_root).as_posix(),
                    "latest_run_summary": (scenario_dir / "latest_run_summary.json")
                    .relative_to(repo_root)
                    .as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return HierarchicalRunPaths(
        run_id=resolved_run_id,
        task=task,
        scenario_dir=scenario_dir,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        latest_run_path=latest_run_path,
    )


def latest_run_id(
    *,
    config: dict[str, Any],
    repo_root: Path,
    task: str,
    alpha: float,
    clients: int,
) -> str:
    latest_path = scenario_run_dir(config, repo_root, task=task, alpha=alpha, clients=clients) / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest P6 Flower run found at {latest_path}. Start the server first or pass --run-id."
        )
    return str(json.loads(latest_path.read_text(encoding="utf-8"))["run_id"])
