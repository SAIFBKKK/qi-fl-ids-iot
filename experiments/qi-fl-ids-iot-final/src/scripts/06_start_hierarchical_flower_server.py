"""Manual P6 hierarchical Flower server launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start manual P6 hierarchical Flower server")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l2", "l3", "l2_family", "l3_attack_type"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-label", default="manual")
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_hierarchical.data import load_hierarchical_config
    from fl_hierarchical.legacy_server import start_legacy_server
    from fl_hierarchical.runtime import configured_address, make_run_id

    args = parse_args()
    config = load_hierarchical_config(args.config)
    address = configured_address(config, args.address)
    run_id = args.run_id or make_run_id()
    print(f"P6 manual Flower server starting | task={args.task} address={address} run_id={run_id}", flush=True)
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
        run_id=run_id,
        runtime_mode=str(args.runtime_label),
    )
    print(f"P6 manual Flower server finished | run_id={run_id}", flush=True)
    print(f"best_round={summary['training']['best_round']} test_macro_f1={summary['test']['metrics']['macro_f1']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
