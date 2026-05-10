"""Run P8.1.5 QGA profile/seed calibration sweep.

This script runs standalone QGA only. It uses train/validation data and never
loads the global test holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.calibration import build_profile_params, calibration_seeds, get_qga_profiles, mask_id, summarize_mask_stability
from qga.config import load_config, make_run_id, relative_to_repo, repo_path, write_json
from qga.data import load_l1_npz, sample_arrays
from qga.fast_eval import evaluate_mask_fast_mlp
from qga.feature_mask import load_feature_names, mask_payload, selected_feature_names, selected_indices
from qga.plotting import plot_qga_figures
from qga.qga_optimizer import run_qga
from qga.report_builder import write_csv


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_one(
    *,
    config: dict,
    profile_name: str,
    seed: int,
    population_size: int | None,
    generations: int | None,
    max_samples_for_fitness: int | None,
) -> dict:
    params = build_profile_params(
        config,
        profile_name=profile_name,
        seed=seed,
        population_size=population_size,
        generations=generations,
        max_samples_for_fitness=max_samples_for_fitness,
    )
    current_mask_id = mask_id(profile_name, seed)
    run_id = make_run_id()
    base_dir = repo_path(config, "outputs.qga_dir") / "calibration" / current_mask_id / "runs" / run_id
    artifacts_dir = base_dir / "artifacts"
    figures_dir = repo_path(config, "outputs.figures_dir") / "calibration" / current_mask_id / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(repo_path(config, "inputs.feature_names"))
    max_samples = int(params["max_samples_for_fitness"])
    train = sample_arrays(load_l1_npz(repo_path(config, "inputs.train_npz")), max_samples=max_samples, seed=seed)
    val = sample_arrays(load_l1_npz(repo_path(config, "inputs.val_npz")), max_samples=max_samples, seed=seed + 1)

    def evaluate(mask, eval_seed):
        return evaluate_mask_fast_mlp(
            mask,
            train,
            val,
            seed=eval_seed,
            batch_size=512,
            epochs=1,
        )

    result = run_qga(n_features=len(feature_names), params=params, evaluate_mask=evaluate)
    selected = selected_feature_names(result.best_mask, feature_names)
    indices = selected_indices(result.best_mask)
    feature_mask_payload = mask_payload(
        result.best_mask,
        feature_names,
        run_id=run_id,
        method=f"P8.1.5 QGA calibration profile={profile_name}",
    )
    feature_mask_payload.update({"mask_id": current_mask_id, "profile": profile_name, "seed": int(seed)})
    ranking_rows = []
    for rank, row in enumerate(result.feature_ranking, start=1):
        enriched = dict(row)
        enriched["rank"] = rank
        enriched["feature_name"] = feature_names[int(row["feature_index"])]
        ranking_rows.append(enriched)

    write_json(artifacts_dir / "qga_config.json", params)
    write_json(
        artifacts_dir / "selected_features.json",
        {
            **feature_mask_payload,
            "best_fitness": result.best_fitness,
            "validation_metrics": result.best_metrics,
            "test_used_for_selection": False,
        },
    )
    write_json(artifacts_dir / "feature_mask.json", feature_mask_payload)
    write_csv(artifacts_dir / "feature_ranking.csv", ranking_rows)
    write_csv(artifacts_dir / "qga_history.csv", result.history)
    write_json(artifacts_dir / "validation_metrics_best_mask.json", result.best_metrics)
    figures = plot_qga_figures(
        history=result.history,
        ranking=ranking_rows,
        mask=result.best_mask,
        feature_names=feature_names,
        figures_dir=figures_dir,
    )
    summary = {
        "accepted": True,
        "phase": "P8.1.5",
        "mask_id": current_mask_id,
        "profile": profile_name,
        "seed": int(seed),
        "run_id": run_id,
        "features_count": len(indices),
        "selected_indices": indices,
        "selected_features": selected,
        "validation": result.best_metrics,
        "fitness": result.best_fitness,
        "test_used_for_selection": False,
        "artifacts": [relative_to_repo(path, config) for path in artifacts_dir.iterdir() if path.is_file()],
        "figures": [relative_to_repo(path, config) for path in figures],
    }
    write_json(artifacts_dir / "run_summary.json", summary)
    latest_summary = base_dir.parents[1] / "latest_run_summary.json"
    write_json(latest_summary, summary)
    return {
        "profile": profile_name,
        "seed": int(seed),
        "mask_id": current_mask_id,
        "run_id": run_id,
        "features_count": len(indices),
        "selected_features": ";".join(selected),
        "selected_indices": " ".join(str(index) for index in indices),
        "validation_macro_f1": result.best_metrics.get("macro_f1"),
        "validation_attack_recall": result.best_metrics.get("recall_attack"),
        "validation_fpr": result.best_metrics.get("FPR"),
        "fitness": result.best_fitness,
        "feature_reduction_ratio": 1 - len(indices) / 28.0,
        "accepted": True,
        "feature_mask_path": relative_to_repo(artifacts_dir / "feature_mask.json", config),
        "run_summary_path": relative_to_repo(artifacts_dir / "run_summary.json", config),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profiles", default=None, help="Comma-separated subset of profiles")
    parser.add_argument("--seeds", default=None, help="Comma-separated subset of seeds")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--max-samples-for-fitness", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    profiles = _parse_csv(args.profiles) or list(get_qga_profiles(config).keys())
    seeds = [int(seed) for seed in (_parse_csv(args.seeds) or calibration_seeds(config))]
    rows = [
        _run_one(
            config=config,
            profile_name=profile,
            seed=seed,
            population_size=args.population_size,
            generations=args.generations,
            max_samples_for_fitness=args.max_samples_for_fitness,
        )
        for profile in profiles
        for seed in seeds
    ]

    reports_dir = repo_path(config, "outputs.reports_dir")
    write_csv(reports_dir / "p8_qga_profile_sweep_summary.csv", rows)
    write_json(reports_dir / "p8_qga_profile_sweep_summary.json", rows)
    feature_names = load_feature_names(repo_path(config, "inputs.feature_names"))
    write_json(
        reports_dir / "p8_qga_mask_stability.json",
        summarize_mask_stability(rows, feature_names=feature_names),
    )
    print(f"P8.1.5 QGA profile sweep completed | runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
