"""QGA L2 profile sweep implementation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from fl_hierarchical.data import load_task_spec
from qga_l2.calibration import get_profiles, get_seeds, mask_id, random_mask
from qga_l2.config import rel, repo_path, write_json
from qga_l2.fast_eval_l2 import train_fast_mlp_l2
from qga_l2.feature_mask import load_feature_names, mask_payload, apply_feature_mask
from qga_l2.fitness_l2 import compute_l2_fitness
from qga_l2.data import load_masked_global_arrays


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_profile_sweep(
    config: dict[str, Any],
    repo_root: Path,
    *,
    profiles: list[str] | None = None,
    seeds: list[int] | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    max_samples_for_fitness: int | None = None,
) -> list[dict[str, Any]]:
    all_profiles = get_profiles(config)
    selected_profiles = profiles or list(all_profiles)
    selected_seeds = seeds or get_seeds(config)
    feature_names = load_feature_names(repo_path(config, "inputs.feature_names"))
    qga_cfg = config["qga_l2"]
    pop = int(population_size or qga_cfg["population_size"])
    gens = int(generations or qga_cfg.get("calibration_generations", qga_cfg["generations"]))
    max_samples = int(max_samples_for_fitness or qga_cfg["max_samples_for_fitness"])
    task_spec = load_task_spec(config, repo_root, "l2")
    rows: list[dict[str, Any]] = []
    for profile_name in selected_profiles:
        profile = all_profiles[profile_name]
        for seed in selected_seeds:
            rng = np.random.default_rng(int(seed))
            best: dict[str, Any] | None = None
            history: list[dict[str, Any]] = []
            for generation in range(gens):
                for individual in range(pop):
                    mask = random_mask(
                        rng,
                        min_features=int(profile["min_features"]),
                        max_features=int(profile["max_features"]),
                        total_features=len(feature_names),
                    )
                    train = load_masked_global_arrays(config, repo_root, split="train", mask=mask, max_samples=max_samples, seed=int(seed))
                    val = load_masked_global_arrays(config, repo_root, split="val", mask=mask, max_samples=max_samples, seed=int(seed) + 101)
                    metrics = train_fast_mlp_l2(
                        train,
                        val,
                        input_dim=int(mask.sum()),
                        output_dim=task_spec.output_dim,
                        seed=int(seed) + generation + individual,
                        max_samples=max_samples,
                        epochs=1,
                    )
                    fitness = compute_l2_fitness(metrics, int(mask.sum()), len(feature_names), profile)
                    item = {
                        "generation": generation,
                        "individual": individual,
                        "features_count": int(mask.sum()),
                        "macro_f1": metrics["macro_f1"],
                        "macro_recall": metrics["macro_recall"],
                        "macro_fpr": metrics["macro_fpr"],
                        "fitness": fitness,
                        "mask": mask,
                    }
                    history.append({key: value for key, value in item.items() if key != "mask"})
                    if best is None or fitness > float(best["fitness"]):
                        best = item
            assert best is not None
            current_mask_id = mask_id(profile_name, int(seed))
            run_id = f"run_{current_mask_id}"
            run_dir = repo_path(config, "outputs.qga_l2_dir") / "calibration" / current_mask_id / "runs" / run_id
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            payload = mask_payload(best["mask"], feature_names, mask_id=current_mask_id, profile=profile_name, seed=int(seed), method="P8-b QGA L2 calibration")
            write_json(artifacts / "feature_mask.json", payload)
            write_json(artifacts / "selected_features.json", payload)
            write_json(artifacts / "validation_metrics_best_mask.json", {k: best[k] for k in ["macro_f1", "macro_recall", "macro_fpr"]})
            write_json(artifacts / "fitness_best.json", {"fitness": best["fitness"], "profile": profile})
            write_json(artifacts / "qga_l2_config.json", {"profile": profile_name, "seed": int(seed), "population_size": pop, "generations": gens})
            _write_csv(artifacts / "qga_history.csv", history)
            summary = {
                "accepted": True,
                "phase": "P8-b",
                "mode": "profile_sweep",
                "mask_id": current_mask_id,
                "profile": profile_name,
                "seed": int(seed),
                "features_count": int(best["features_count"]),
                "validation": {k: best[k] for k in ["macro_f1", "macro_recall", "macro_fpr"]},
                "fitness": float(best["fitness"]),
                "test_used_for_selection": False,
                "feature_mask_path": rel(artifacts / "feature_mask.json", repo_root),
            }
            write_json(artifacts / "run_summary.json", summary)
            rows.append(
                {
                    "profile": profile_name,
                    "seed": int(seed),
                    "mask_id": current_mask_id,
                    "features_count": int(best["features_count"]),
                    "selected_features": ";".join(payload["selected_features"]),
                    "validation_macro_f1": best["macro_f1"],
                    "validation_macro_recall": best["macro_recall"],
                    "validation_macro_fpr": best["macro_fpr"],
                    "fitness": best["fitness"],
                    "feature_reduction_ratio": 1 - int(best["features_count"]) / 28.0,
                    "accepted": True,
                    "feature_mask_path": rel(artifacts / "feature_mask.json", repo_root),
                    "run_summary_path": rel(artifacts / "run_summary.json", repo_root),
                }
            )
    reports = repo_path(config, "outputs.reports_dir")
    _write_csv(reports / "p8b_qga_l2_profile_sweep_summary.csv", rows)
    write_json(reports / "p8b_qga_l2_profile_sweep_summary.json", rows)
    return rows
