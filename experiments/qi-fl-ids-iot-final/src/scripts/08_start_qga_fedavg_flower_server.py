"""Start the true Flower server for P8 FedAvg + QGA L1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config
from qga.flower_runtime import start_qga_flower_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P8 FedAvg+QGA true Flower server")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-label", default="manual")
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--mask-source", choices=["final_selected_mask", "latest_qga_run"], default="final_selected_mask")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    address = args.address or str(config.get("flower", {}).get("address", "127.0.0.1:8083"))
    summary = start_qga_flower_server(
        config=config,
        repo_root=Path.cwd().resolve(),
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        mode=args.mode,
        address=address,
        run_id=args.run_id,
        runtime_mode=args.runtime_label,
        mask_path=args.mask_path,
        mask_source=args.mask_source,
        evaluate_test=not args.validation_only,
    )
    print(f"P8 FedAvg+QGA Flower server finished | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
