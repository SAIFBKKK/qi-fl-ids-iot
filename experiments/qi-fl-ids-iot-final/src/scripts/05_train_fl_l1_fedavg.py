"""CLI for P5 FedAvg L1 training modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P5 FedAvg L1")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=["verify", "smoke", "full", "grid"], default="verify")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_l1.fedavg_server import resolve_sample_limit_for_mode, run_fedavg_scenario
    from fl_l1.scenario_loader import list_expected_scenarios, load_config, load_l1_scenario
    from fl_l1.verify_setup import verify_setup

    args = parse_args()
    if args.mode == "verify":
        summary = verify_setup(args.config, write_outputs=True)
        print("P5 FedAvg L1 verify complete")
        print(f"accepted: {summary['accepted']}")
        return 0 if summary["accepted"] else 1

    config = load_config(args.config)
    repo_root = Path.cwd().resolve()
    smoke_default = config.get("execution", {}).get("smoke_max_samples_per_client")
    if args.mode == "grid":
        scenarios = list_expected_scenarios(config)
        rounds = int(args.rounds or config["federated"]["rounds"])
        max_samples = resolve_sample_limit_for_mode(
            mode=args.mode,
            requested_max_samples=args.max_samples_per_client,
            default_smoke_max_samples=smoke_default,
        )
    else:
        alpha = float(args.alpha if args.alpha is not None else config["scenario"]["default_alpha"])
        clients = int(args.clients if args.clients is not None else config["scenario"]["default_k"])
        scenarios = [(alpha, clients)]
        rounds = int(
            args.rounds
            if args.rounds is not None
            else (config["execution"]["smoke_rounds"] if args.mode == "smoke" else config["federated"]["rounds"])
        )
        max_samples = resolve_sample_limit_for_mode(
            mode=args.mode,
            requested_max_samples=args.max_samples_per_client,
            default_smoke_max_samples=smoke_default,
        )

    if args.mode != "smoke" and args.max_samples_per_client is not None:
        print("Ignoring --max-samples-per-client outside smoke mode; full/grid use all client samples.")

    summaries = []
    for alpha, clients in scenarios:
        scenario = load_l1_scenario(config, repo_root, alpha=alpha, num_clients=clients)
        summaries.append(
            run_fedavg_scenario(
                config=config,
                repo_root=repo_root,
                scenario=scenario,
                mode=args.mode,
                rounds=rounds,
                max_samples_per_client=max_samples,
            )
        )

    print(f"P5 FedAvg L1 {args.mode} complete")
    print(f"scenarios: {len(summaries)}")
    for summary in summaries:
        print(
            f"- alpha={summary['alpha']} k={summary['num_clients']} "
            f"rounds={summary['rounds']} best_round={summary['best_round']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
