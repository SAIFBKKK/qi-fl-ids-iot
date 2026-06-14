"""Runtime helpers for P5.2/P5.2.1 Flower executions."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fl_l1.scenario_loader import repo_path
from fl_l1_flower.data import scenario_run_dir


@dataclass(frozen=True)
class FlowerRunPaths:
    """Resolved run-specific directories."""

    run_id: str
    scenario_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    latest_run_path: Path


def make_run_id() -> str:
    """Create a deterministic-format run id for one Flower execution."""

    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_address(address: str) -> tuple[str, int]:
    """Parse `host:port` Flower addresses."""

    if ":" not in address:
        raise ValueError(f"Flower address must be host:port, got {address!r}")
    host, raw_port = address.rsplit(":", 1)
    if not host:
        host = "127.0.0.1"
    return host, int(raw_port)


def is_port_available(address: str) -> bool:
    """Return true if a server can bind the configured address."""

    host, port = parse_address(address)
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "[::]", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def assert_port_available(address: str) -> None:
    """Raise a clear error if the Flower server port is already occupied."""

    if not is_port_available(address):
        host, port = parse_address(address)
        raise RuntimeError(
            f"Flower server port is already in use: {host}:{port}. "
            f"On Windows, inspect it with: netstat -ano | findstr :{port}"
        )


def prepare_run_paths(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    run_id: str | None = None,
    mark_latest: bool = True,
) -> FlowerRunPaths:
    """Create run-specific directories under `runs/{run_id}`."""

    resolved_run_id = run_id or make_run_id()
    scenario_dir = scenario_run_dir(config, repo_root, alpha=alpha, clients=clients)
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
                    "run_dir": run_dir.relative_to(repo_root).as_posix(),
                    "logs_dir": logs_dir.relative_to(repo_root).as_posix(),
                    "latest_run_summary": (latest_run_path.parent / "latest_run_summary.json")
                    .relative_to(repo_root)
                    .as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return FlowerRunPaths(
        run_id=resolved_run_id,
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
    alpha: float,
    clients: int,
) -> str:
    """Read the latest server-created run id for manual clients."""

    scenario_dir = scenario_run_dir(config, repo_root, alpha=alpha, clients=clients)
    latest_path = scenario_dir / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest Flower run found at {latest_path}. "
            "Start the server first or pass --run-id explicitly."
        )
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    return str(payload["run_id"])


def configured_address(config: dict[str, Any], override: str | None = None) -> str:
    """Resolve Flower address from CLI or config."""

    return str(override or config.get("flower", {}).get("address", "127.0.0.1:8080"))


def artifact_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repo-relative artifact path."""

    return repo_path(repo_root, relative_path)
