"""Run true-Flower short validation for calibrated QGA masks.

The runs are validation-only: the global test holdout is not loaded and is not
used for mask ranking.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config, load_json, repo_path, write_json
from qga.flower_runtime import run_qga_flower_smoke_subprocess
from qga.report_builder import write_csv


def _load_candidates(config: dict, *, top_n: int) -> list[dict]:
    path = repo_path(config, "outputs.reports_dir") / "p8_qga_profile_sweep_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"profile sweep summary not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=lambda row: float(row.get("fitness", 0.0)), reverse=True)
    return rows[: int(top_n)]


def _scenario_label(alpha: float, clients: int) -> str:
    return f"alpha={alpha}_k={clients}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    parser.add_argument("--address", default="127.0.0.1:8083")
    parser.add_argument("--only-first", action="store_true", help="Run only one candidate/scenario for a lightweight validation")
    args = parser.parse_args()
    config = load_config(args.config)
    candidates = _load_candidates(config, top_n=args.top_n)
    scenarios = list(config["qga_calibration"]["primary_scenarios"])
    rows: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        for scenario_index, scenario in enumerate(scenarios):
            if args.only_first and (candidate_index > 0 or scenario_index > 0):
                continue
            alpha = float(scenario["alpha"])
            clients = int(scenario["clients"])
            mask_path = repo_path(config, None) / candidate["feature_mask_path"]
            summary = run_qga_flower_smoke_subprocess(
                config_path=Path(args.config),
                alpha=alpha,
                clients=clients,
                rounds=int(args.rounds),
                max_samples_per_client=int(args.max_samples_per_client),
                address=args.address,
                mask_path=mask_path,
                evaluate_test=False,
                mode="smoke",
            )
            val_metrics = summary["validation"]["metrics"]
            rows.append(
                {
                    "mask_id": candidate["mask_id"],
                    "profile": candidate["profile"],
                    "seed": int(candidate["seed"]),
                    "alpha": alpha,
                    "clients": clients,
                    "scenario": _scenario_label(alpha, clients),
                    "rounds": int(args.rounds),
                    "val_macro_f1": val_metrics.get("macro_f1"),
                    "val_attack_recall": val_metrics.get("recall_attack"),
                    "val_fpr": val_metrics.get("FPR"),
                    "features_count": int(candidate["features_count"]),
                    "bandwidth_total_bytes": summary["communication"].get("communication_cumulative_bytes"),
                    "model_size_bytes": summary["communication"].get("model_size_bytes"),
                    "true_flower_runtime": summary.get("true_flower_runtime") is True,
                    "test_sent_to_clients": summary["dataset"].get("test_sent_to_clients"),
                    "test_evaluated": summary["test"].get("evaluated"),
                    "accepted": summary.get("accepted"),
                    "run_id": summary.get("run_id"),
                    "run_summary": summary.get("artifacts", [""])[0],
                }
            )
    reports_dir = repo_path(config, "outputs.reports_dir")
    existing_path = reports_dir / "p8_qga_flower_short_validation.csv"
    existing: list[dict] = []
    if existing_path.exists():
        with existing_path.open("r", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
    merged = existing + rows
    write_csv(existing_path, merged)
    write_json(reports_dir / "p8_qga_flower_short_validation.json", merged)
    print(f"P8.1.5 Flower short validation completed | new_runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
