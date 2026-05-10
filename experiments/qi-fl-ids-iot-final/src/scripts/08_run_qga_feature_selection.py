"""Run P8 QGA feature selection on L1 train/validation data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config, make_run_id, qga_params, relative_to_repo, repo_path, write_json
from qga.data import load_qga_train_val
from qga.fast_eval import evaluate_mask_fast_mlp
from qga.feature_mask import load_feature_names, mask_payload, selected_feature_names, selected_indices
from qga.plotting import plot_qga_figures
from qga.qga_optimizer import run_qga
from qga.report_builder import build_markdown_report, build_qga_run_summary, write_csv, write_latest_pointers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    params = qga_params(config, mode=args.mode)
    run_id = make_run_id()
    qga_dir = repo_path(config, "outputs.qga_dir")
    run_dir = qga_dir / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    figures_dir = repo_path(config, "outputs.figures_dir") / run_id
    for path in [artifacts_dir, logs_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(repo_path(config, "inputs.feature_names"))
    train, val = load_qga_train_val(config, mode=args.mode)
    log_path = logs_dir / "qga_run.log"
    log_path.write_text(
        f"P8 QGA started | mode={args.mode} train={train.rows} val={val.rows} test_used_for_selection=false\n",
        encoding="utf-8",
    )

    def evaluate(mask, seed):
        return evaluate_mask_fast_mlp(
            mask,
            train,
            val,
            seed=seed,
            batch_size=512,
            epochs=1 if args.mode == "smoke" else 2,
        )

    result = run_qga(n_features=len(feature_names), params=params, evaluate_mask=evaluate)
    selected = selected_feature_names(result.best_mask, feature_names)
    indices = selected_indices(result.best_mask)

    feature_mask_payload = mask_payload(
        result.best_mask,
        feature_names,
        run_id=run_id,
        method="QGA theta-vector bounded-mask L1",
    )
    selected_payload = {
        **feature_mask_payload,
        "best_fitness": result.best_fitness,
        "validation_metrics": result.best_metrics,
        "test_used_for_selection": False,
    }
    ranking_rows = []
    for rank, row in enumerate(result.feature_ranking, start=1):
        enriched = dict(row)
        enriched["rank"] = rank
        enriched["feature_name"] = feature_names[int(row["feature_index"])]
        ranking_rows.append(enriched)

    write_json(artifacts_dir / "qga_config.json", params)
    write_json(artifacts_dir / "selected_features.json", selected_payload)
    write_json(artifacts_dir / "feature_mask.json", feature_mask_payload)
    write_csv(artifacts_dir / "feature_ranking.csv", ranking_rows)
    write_csv(artifacts_dir / "qga_history.csv", result.history)
    write_json(artifacts_dir / "fitness_best.json", {"best_fitness": result.best_fitness, "best_metrics": result.best_metrics})
    write_json(artifacts_dir / "fitness_weights.json", params["weights"])
    write_json(artifacts_dir / "validation_metrics_best_mask.json", result.best_metrics)

    figure_paths = plot_qga_figures(
        history=result.history,
        ranking=ranking_rows,
        mask=result.best_mask,
        feature_names=feature_names,
        figures_dir=figures_dir,
    )
    run_summary_path = artifacts_dir / "run_summary.json"
    artifacts = [relative_to_repo(path, config) for path in artifacts_dir.iterdir() if path.is_file()]
    artifacts.append(relative_to_repo(run_summary_path, config))
    figures = [relative_to_repo(path, config) for path in figure_paths]
    summary = build_qga_run_summary(
        config=config,
        run_id=run_id,
        mode=args.mode,
        params=params,
        selected_features=selected,
        selected_indices=indices,
        best_fitness=result.best_fitness,
        best_metrics=result.best_metrics,
        artifacts=artifacts,
        figures=figures,
    )
    write_json(run_summary_path, summary)
    write_latest_pointers(config=config, run_id=run_id, run_dir=run_dir, qga_dir=qga_dir, summary=summary)
    build_markdown_report(repo_path(config, "outputs.reports_dir") / "p8_qga_latest_report.md", summary)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"P8 QGA finished | selected={len(selected)} fitness={result.best_fitness:.6f}\n")
    print(f"P8 QGA run completed | accepted={summary['accepted']} | run_id={run_id}")
    print(f"Selected features ({len(selected)}): {', '.join(selected)}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
