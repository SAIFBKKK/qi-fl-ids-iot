"""Run P6 hierarchical Flower smoke/full through subprocess legacy runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P6 hierarchical Flower")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l2", "l3", "l2_family", "l3_attack_type"])
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--address", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_hierarchical.data import load_hierarchical_config
    from fl_hierarchical.legacy_server import run_legacy_local_smoke
    from fl_hierarchical.runtime import configured_address

    args = parse_args()
    config = load_hierarchical_config(args.config)
    address = configured_address(config, args.address)
    max_samples = args.max_samples_per_client if args.mode == "smoke" else None
    if args.mode != "smoke" and args.max_samples_per_client is not None:
        print("Ignoring --max-samples-per-client outside smoke mode; full uses all client samples.")
    summary = run_legacy_local_smoke(
        config_path=args.config,
        task=args.task,
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        max_samples_per_client=max_samples,
        mode=args.mode,
        server_address=address,
        run_id=args.run_id,
    )
    print(f"P6 Flower {args.task} {args.mode} complete")
    print(f"run_id={summary['run_id']} accepted={summary['accepted']}")
    print(f"best_round={summary['training']['best_round']} test_macro_f1={summary['test']['metrics']['macro_f1']:.4f}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
