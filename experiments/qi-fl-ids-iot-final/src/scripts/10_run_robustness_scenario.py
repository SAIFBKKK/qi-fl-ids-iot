"""Run one P10 robustness scenario manually."""

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
    parser.add_argument("--method", required=True, choices=["fedavg", "fedavg_qga", "qifa", "qifa_qga"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--attack-type", required=True, choices=["clean", "label_flip", "attack_to_normal", "feature_noise"])
    parser.add_argument("--poison-rate", type=float, default=0.2)
    parser.add_argument("--poisoned-clients", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    _bootstrap()
    from robustness.config import load_config
    from robustness.runner import run_robustness_scenario
    from robustness.scenario import load_scenario

    config = load_config(args.config)
    scenario = load_scenario(
        config,
        alpha=args.alpha,
        clients=args.clients,
        rounds=args.rounds,
        attack_type=args.attack_type,
        poison_rate=args.poison_rate,
        poisoned_clients=args.poisoned_clients,
        method=args.method,
    )
    summary = run_robustness_scenario(config=config, scenario=scenario, mode="manual", max_samples=args.max_samples)
    print(f"P10 scenario complete run_id={summary['run_id']} accepted={summary['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
