"""Manual P6 hierarchical Flower client launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start manual P6 hierarchical Flower client")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["l2", "l3", "l2_family", "l3_attack_type"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--address", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_hierarchical.data import load_hierarchical_config
    from fl_hierarchical.legacy_client import start_legacy_client
    from fl_hierarchical.runtime import configured_address

    args = parse_args()
    config = load_hierarchical_config(args.config)
    address = configured_address(config, args.address)
    print(f"P6 manual Flower client starting | task={args.task} client={args.client_id} address={address}", flush=True)
    start_legacy_client(
        config=config,
        repo_root=Path.cwd().resolve(),
        task=args.task,
        client_id=args.client_id,
        alpha=float(args.alpha),
        clients=int(args.clients),
        server_address=address,
        max_samples_per_client=args.max_samples_per_client if args.mode == "smoke" else None,
        run_id=args.run_id,
    )
    print(f"P6 manual Flower client finished | client={args.client_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
