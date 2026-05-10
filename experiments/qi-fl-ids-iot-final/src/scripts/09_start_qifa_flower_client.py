"""Start one true Flower client for P9 QIFA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.config import load_config
from qifa.data import configured_address
from qifa.flower_runtime import start_qifa_client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--use-qga-mask", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    address = configured_address(config, args.address)
    start_qifa_client(
        config=config,
        repo_root=Path.cwd().resolve(),
        client_id=str(args.client_id),
        alpha=float(args.alpha),
        clients=int(args.clients),
        variant=str(args.variant),
        gamma=float(args.gamma),
        address=address,
        mode=args.mode,
        run_id=args.run_id,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        use_qga_mask=bool(args.use_qga_mask),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
