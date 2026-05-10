"""Start P8-b QGA L2 Flower server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config
from qga_l2.flower_runtime import start_server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = start_server(
        config,
        alpha=args.alpha,
        clients=args.clients,
        rounds=args.rounds,
        address=args.address or config["flower"]["address"],
        mode=args.mode,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        run_id=args.run_id,
        mask_path=args.mask_path,
        evaluate_test=not args.validation_only,
    )
    print(f"P8-b QGA L2 Flower server finished | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
