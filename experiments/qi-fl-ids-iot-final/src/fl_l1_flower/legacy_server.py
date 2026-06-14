"""Legacy Flower server fallback for P5.2.

This path starts a real Flower gRPC server and real Flower NumPyClients on
localhost. It is intended for reliable smoke validation when Ray simulation is
too heavy on Windows, while keeping the ClientApp/ServerApp code path present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.server import ServerConfig

from fl_l1_flower.data import (
    concatenate_validation_arrays,
    load_flower_config,
    load_scenario,
)
from fl_l1_flower.runtime import assert_port_available, configured_address, prepare_run_paths
from fl_l1_flower.strategy import FlowerL1FedAvgStrategy


def build_legacy_strategy(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    run_id: str | None = None,
    runtime_mode: str = "legacy-local",
) -> FlowerL1FedAvgStrategy:
    """Build the shared strategy for legacy Flower server execution."""

    config["scenario"]["alpha"] = float(alpha)
    config["scenario"]["clients"] = int(clients)
    config["scenario"]["rounds"] = int(rounds)
    scenario = load_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        run_id=run_id,
    )
    validation_arrays = concatenate_validation_arrays(
        scenario,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        seed=int(config["training"]["seed"]),
    )
    return FlowerL1FedAvgStrategy(
        config=config,
        repo_root=repo_root,
        scenario=scenario,
        run_dir=run_paths.run_dir,
        validation_arrays=validation_arrays,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        mode=mode,
        run_id=run_paths.run_id,
        runtime_mode=runtime_mode,
    )


def start_legacy_server(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    server_address: str,
    run_id: str | None = None,
    check_port: bool = True,
    runtime_mode: str = "manual",
) -> dict[str, Any]:
    """Start a real Flower server and block until the configured rounds finish."""

    if check_port:
        assert_port_available(server_address)
    strategy = build_legacy_strategy(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        rounds=rounds,
        max_samples_per_client=max_samples_per_client,
        mode=mode,
        run_id=run_id,
        runtime_mode=runtime_mode,
    )
    strategy._server_log(f"server starting | address={server_address} run_id={strategy.run_id}")
    strategy._server_log(f"server listening on address={server_address}")
    strategy._server_log(
        f"server waiting for clients | min_available={config['flower']['min_available_clients']}"
    )
    fl.server.start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )
    summary = strategy.finalize()
    strategy._server_log("server finished")
    return summary


def run_legacy_local_smoke(
    *,
    config_path: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    server_address: str = "127.0.0.1:8095",
    run_id: str | None = None,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    """Run server + clients as separate subprocesses through real Flower."""

    repo_root = Path.cwd().resolve()
    config = load_flower_config(config_path)
    assert_port_available(server_address)
    scenario = load_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        run_id=run_id,
    )
    scripts_dir = repo_root / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts"
    server_script = scripts_dir / "05_2_start_flower_server.py"
    client_script = scripts_dir / "05_2_start_flower_client.py"
    common = [
        "--config",
        str(config_path),
        "--alpha",
        str(alpha),
        "--clients",
        str(clients),
        "--address",
        server_address,
        "--run-id",
        run_paths.run_id,
    ]
    server_cmd = [
        sys.executable,
        str(server_script),
        *common,
        "--rounds",
        str(rounds),
        "--mode",
        mode,
        "--runtime-label",
        "subprocess",
    ]
    if mode == "smoke" and max_samples_per_client is not None:
        server_cmd.extend(["--max-samples-per-client", str(max_samples_per_client)])

    processes: list[tuple[str, subprocess.Popen]] = []
    opened_files = []

    def start_process(name: str, command: list[str]) -> None:
        log_path = run_paths.logs_dir / f"{name}_stdout.log"
        handle = log_path.open("w", encoding="utf-8")
        opened_files.append(handle)
        processes.append(
            (
                name,
                subprocess.Popen(
                    command,
                    cwd=str(repo_root),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
        )

    try:
        start_process("server", server_cmd)
        time.sleep(5.0)
        for partition in scenario.clients:
            client_cmd = [
                sys.executable,
                str(client_script),
                *common,
                "--client-id",
                partition.client_id,
                "--mode",
                mode,
            ]
            if mode == "smoke" and max_samples_per_client is not None:
                client_cmd.extend(["--max-samples-per-client", str(max_samples_per_client)])
            start_process(partition.client_id, client_cmd)

        deadline = time.monotonic() + float(timeout_sec)
        for name, process in processes:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"{name} did not finish within {timeout_sec}s") from exc

        failures = [(name, process.returncode) for name, process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"Flower subprocess failure(s): {failures}")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in opened_files:
            handle.close()

    summary_path = run_paths.artifacts_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Flower subprocess run did not produce {summary_path}")
    import json

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P5.2 legacy Flower L1 server")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--address", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-label", default="manual")
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    args = parse_args()
    config = load_flower_config(args.config)
    address = configured_address(config, args.address or args.server_address)
    summary = start_legacy_server(
        config=config,
        repo_root=Path.cwd().resolve(),
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        mode=args.mode,
        server_address=address,
        run_id=args.run_id,
        runtime_mode=args.runtime_label,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
