"""Start one true Flower client for P8 FedAvg + QGA L1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config
from qga.flower_runtime import start_qga_flower_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P8 FedAvg+QGA true Flower client")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--address", default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--mask-source", choices=["final_selected_mask", "latest_qga_run"], default="final_selected_mask")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    address = args.address or str(config.get("flower", {}).get("address", "127.0.0.1:8083"))
    start_qga_flower_client(
        config=config,
        repo_root=Path.cwd().resolve(),
        client_id=args.client_id,
        alpha=float(args.alpha),
        clients=int(args.clients),
        address=address,
        max_samples_per_client=args.max_samples_per_client,
        run_id=args.run_id,
        mode=args.mode,
        mask_path=args.mask_path,
        mask_source=args.mask_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
