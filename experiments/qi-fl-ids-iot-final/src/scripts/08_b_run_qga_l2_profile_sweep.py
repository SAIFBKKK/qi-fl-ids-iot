"""Run P8-b QGA L2 profile sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config, repo_path
from qga_l2.profile_sweep import run_profile_sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--max-samples-for-fitness", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = run_profile_sweep(
        config,
        repo_path(config),
        profiles=args.profiles,
        seeds=args.seeds,
        population_size=args.population_size,
        generations=args.generations,
        max_samples_for_fitness=args.max_samples_for_fitness,
    )
    print(f"P8-b QGA L2 profile sweep completed | runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
