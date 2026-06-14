"""Start the true Flower server for P9 QIFA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.config import load_config
from qifa.data import configured_address
from qifa.flower_runtime import start_qifa_server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-label", default="manual")
    parser.add_argument("--use-qga-mask", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    address = configured_address(config, args.address)
    summary = start_qifa_server(
        config=config,
        repo_root=Path.cwd().resolve(),
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        variant=str(args.variant),
        gamma=float(args.gamma),
        address=address,
        mode=args.mode,
        run_id=args.run_id,
        runtime_mode=args.runtime_label,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        use_qga_mask=bool(args.use_qga_mask),
    )
    print(f"P9 QIFA Flower server finished | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
