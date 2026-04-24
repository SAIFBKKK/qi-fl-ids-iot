from __future__ import annotations

import argparse

from src.common.paths import DATA_DIR, OUTPUTS_DIR
from src.data.analysis.client_distribution_report import save_client_distribution_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_dir = DATA_DIR / "processed" / args.scenario
    out_dir = OUTPUTS_DIR / "reports" / args.scenario / "per_client"
    out_dir.mkdir(parents=True, exist_ok=True)

    for node_dir in sorted([p for p in scenario_dir.glob("node*") if p.is_dir()]):
        npz_path = node_dir / "train_preprocessed.npz"
        out_json = out_dir / f"{node_dir.name}_report.json"
        save_client_distribution_report(npz_path, out_json)
        print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
