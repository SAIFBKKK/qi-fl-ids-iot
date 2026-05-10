"""Run P5.2 true Flower L1 smoke/full simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P5.2 Flower L1")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--runtime", choices=["legacy-local", "simulation"], default="legacy-local")
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--address", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_l1_flower.data import load_flower_config
    from fl_l1_flower.legacy_server import run_legacy_local_smoke
    from fl_l1_flower.runtime import configured_address
    from fl_l1_flower.server_app import run_flower_l1_simulation

    args = parse_args()
    config = load_flower_config(args.config)
    alpha = float(args.alpha if args.alpha is not None else config["scenario"]["alpha"])
    clients = int(args.clients if args.clients is not None else config["scenario"]["clients"])
    rounds = int(args.rounds if args.rounds is not None else config["scenario"]["rounds"])
    address = configured_address(config, args.address or args.server_address)
    max_samples = args.max_samples_per_client if args.mode == "smoke" else None
    if args.mode != "smoke" and args.max_samples_per_client is not None:
        print("Ignoring --max-samples-per-client outside smoke mode; full uses all client samples.")
    if args.runtime == "simulation":
        summary = run_flower_l1_simulation(
            config_path=args.config,
            alpha=alpha,
            clients=clients,
            rounds=rounds,
            max_samples_per_client=max_samples,
            mode=args.mode,
            run_id=args.run_id,
        )
    else:
        summary = run_legacy_local_smoke(
            config_path=args.config,
            alpha=alpha,
            clients=clients,
            rounds=rounds,
            max_samples_per_client=max_samples,
            mode=args.mode,
            server_address=address,
            run_id=args.run_id,
        )
    print(f"P5.2 Flower L1 {args.mode} complete")
    print(f"runtime={args.runtime}")
    print(f"alpha={summary['alpha']} k={summary['num_clients']} rounds={summary['rounds']}")
    print(f"best_round={summary['best_round']} test_macro_f1={summary['metrics_test']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
