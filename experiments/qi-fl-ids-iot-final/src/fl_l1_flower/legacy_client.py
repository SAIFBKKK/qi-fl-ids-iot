"""Legacy Flower client fallback for P5.2.

This uses the real Flower `start_numpy_client` transport. It is kept separate
from the ClientApp path because Flower 1.8/Ray simulation can be heavy on
Windows workstations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1_flower.client_app import FlowerL1Client
from fl_l1_flower.data import load_flower_config, load_scenario, scenario_run_dir
from fl_l1_flower.runtime import configured_address, latest_run_id, prepare_run_paths


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def start_legacy_client(
    *,
    config: dict[str, Any],
    repo_root: Path,
    client_id: str,
    alpha: float,
    clients: int,
    server_address: str,
    max_samples_per_client: int | None,
    run_id: str | None = None,
) -> None:
    """Start one real Flower NumPyClient."""

    scenario = load_scenario(config, repo_root, alpha=alpha, clients=clients)
    resolved_run_id = run_id or latest_run_id(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
    )
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        run_id=resolved_run_id,
        mark_latest=False,
    )
    _append_log(
        run_paths.logs_dir / "flower_clients.log",
        f"{client_id} process starting | address={server_address} run_id={resolved_run_id}",
    )
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} loading data")
    client = FlowerL1Client(
        client_id=client_id,
        config=config,
        scenario=scenario,
        logs_dir=run_paths.logs_dir,
        max_samples_per_client=max_samples_per_client,
    )
    _append_log(
        run_paths.logs_dir / "flower_clients.log",
        f"{client_id} connecting to server | address={server_address}",
    )
    try:
        fl.client.start_client(server_address=server_address, client=client.to_client())
        _append_log(
            run_paths.logs_dir / "flower_clients.log",
            f"{client_id} connection finished cleanly",
        )
    except BaseException as exc:
        _append_log(
            run_paths.logs_dir / "flower_clients.log",
            f"{client_id} connection failed | error={exc}",
        )
        raise


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one P5.2 legacy Flower L1 client")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--address", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    args = parse_args()
    config = load_flower_config(args.config)
    address = configured_address(config, args.address or args.server_address)
    start_legacy_client(
        config=config,
        repo_root=Path.cwd().resolve(),
        client_id=args.client_id,
        alpha=float(args.alpha),
        clients=int(args.clients),
        server_address=address,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
