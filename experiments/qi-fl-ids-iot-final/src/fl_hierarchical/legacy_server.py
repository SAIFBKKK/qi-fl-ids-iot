"""Legacy/manual Flower server and subprocess launcher for P6."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.server import ServerConfig

from fl_hierarchical.data import (
    concatenate_validation_arrays,
    load_hierarchical_config,
    load_l2_index_scenario,
    load_task_spec,
)
from fl_hierarchical.runtime import assert_port_available, configured_address, prepare_run_paths
from fl_hierarchical.strategy import HierarchicalFedAvgStrategy


def build_legacy_strategy(
    *,
    config: dict[str, Any],
    repo_root: Path,
    task: str,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    run_id: str | None = None,
    runtime_mode: str = "legacy-local",
) -> HierarchicalFedAvgStrategy:
    config["scenario"]["alpha"] = float(alpha)
    config["scenario"]["clients"] = int(clients)
    config["scenario"]["rounds"] = int(rounds)
    task_spec = load_task_spec(config, repo_root, task)
    scenario = load_l2_index_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        task=task_spec.task,
        alpha=alpha,
        clients=clients,
        run_id=run_id,
    )
    validation_arrays = concatenate_validation_arrays(
        config,
        repo_root,
        scenario,
        task_spec,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
    )
    return HierarchicalFedAvgStrategy(
        config=config,
        repo_root=repo_root,
        scenario=scenario,
        task_spec=task_spec,
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
    task: str,
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
    if check_port:
        assert_port_available(server_address)
    strategy = build_legacy_strategy(
        config=config,
        repo_root=repo_root,
        task=task,
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
    strategy._server_log(f"server waiting for clients | min_available={config['flower']['min_available_clients']}")
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
    task: str,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    server_address: str,
    run_id: str | None = None,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_hierarchical_config(config_path)
    assert_port_available(server_address)
    task_spec = load_task_spec(config, repo_root, task)
    scenario = load_l2_index_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_run_paths(
        config=config,
        repo_root=repo_root,
        task=task_spec.task,
        alpha=alpha,
        clients=clients,
        run_id=run_id,
    )
    scripts_dir = repo_root / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts"
    server_script = scripts_dir / "06_start_hierarchical_flower_server.py"
    client_script = scripts_dir / "06_start_hierarchical_flower_client.py"
    common = [
        "--config",
        str(config_path),
        "--task",
        task_spec.short_name,
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
            raise RuntimeError(f"P6 Flower subprocess failure(s): {failures}")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in opened_files:
            handle.close()

    summary_path = run_paths.artifacts_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"P6 Flower subprocess run did not produce {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P6 hierarchical Flower server")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l2", "l3", "l2_family", "l3_attack_type"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--address", default=None)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-label", default="manual")
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    args = parse_args()
    config = load_hierarchical_config(args.config)
    address = configured_address(config, args.address or args.server_address)
    summary = start_legacy_server(
        config=config,
        repo_root=Path.cwd().resolve(),
        task=args.task,
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
