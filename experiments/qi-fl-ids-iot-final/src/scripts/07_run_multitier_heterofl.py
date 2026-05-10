"""Run P7 Multi-tier HeteroFL in-process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P7 Multi-tier HeteroFL")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l1", "l2", "l1_binary", "l2_family"])
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from multitier_heterofl.config import load_config
    from multitier_heterofl.runtime import run_multitier_heterofl

    args = parse_args()
    config = load_config(args.config)
    max_samples = args.max_samples_per_client if args.mode == "smoke" else None
    if args.mode != "smoke" and args.max_samples_per_client is not None:
        print("Ignoring --max-samples-per-client outside smoke mode; full uses all client samples.")
    summary = run_multitier_heterofl(
        config=config,
        repo_root=Path.cwd().resolve(),
        task=args.task,
        mode=args.mode,
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        max_samples_per_client=max_samples,
        run_id=args.run_id,
    )
    print(f"P7 HeteroFL {args.task} {args.mode} complete")
    print(f"run_id={summary['run_id']} accepted={summary['accepted']}")
    print(f"best_round={summary['training']['best_round']} test_macro_f1={summary['test']['metrics']['macro_f1']:.4f}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
