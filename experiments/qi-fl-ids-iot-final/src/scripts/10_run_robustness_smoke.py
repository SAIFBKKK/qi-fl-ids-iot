"""Run a lightweight P10 robustness smoke scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml"))
    parser.add_argument("--method", default="fedavg", choices=["fedavg", "fedavg_qga", "qifa", "qifa_qga"])
    parser.add_argument("--attack-type", default=None)
    parser.add_argument("--poison-rate", type=float, default=None)
    parser.add_argument("--poisoned-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    _bootstrap()
    from robustness.config import load_config
    from robustness.runner import run_robustness_scenario
    from robustness.scenario import load_scenario

    config = load_config(args.config)
    default = config["default_scenario"]
    scenario = load_scenario(
        config,
        alpha=float(default["alpha"]),
        clients=int(default["clients"]),
        rounds=int(args.rounds or config["execution"]["smoke_rounds"]),
        attack_type=str(args.attack_type or default["attack_type"]),
        poison_rate=float(args.poison_rate if args.poison_rate is not None else default["poison_rate"]),
        poisoned_clients=int(args.poisoned_clients or default["poisoned_clients"]),
        method=args.method,
    )
    summary = run_robustness_scenario(
        config=config,
        scenario=scenario,
        mode="smoke",
        max_samples=int(args.max_samples or config["execution"]["smoke_max_samples_per_client"]),
    )
    print(f"P10 smoke complete run_id={summary['run_id']} macro_f1={summary['test']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
