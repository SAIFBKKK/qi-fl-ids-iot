"""Report and artifact builders for P8 QGA."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from qga.config import relative_to_repo, write_json
from qga.summary_schema import accepted_from_criteria, qga_criteria


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0].keys()) if rows else [])
    with out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def build_qga_run_summary(
    *,
    config: dict[str, Any],
    run_id: str,
    mode: str,
    params: dict[str, Any],
    selected_features: list[str],
    selected_indices: list[int],
    best_fitness: float,
    best_metrics: dict[str, Any],
    artifacts: list[str],
    figures: list[str],
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    criteria = qga_criteria(
        selected_features_count=len(selected_indices),
        min_features=int(params["min_features"]),
        max_features=int(params["max_features"]),
    )
    summary = {
        "accepted": accepted_from_criteria(criteria) and not errors,
        "phase": "P8",
        "method": "QGA Feature Selection",
        "task": "l1_binary",
        "mode": mode,
        "run_id": run_id,
        "qga": {
            "population_size": int(params["population_size"]),
            "generations": int(params["generations"]),
            "mutation_rate": float(params["mutation_rate"]),
            "min_features": int(params["min_features"]),
            "max_features": int(params["max_features"]),
            "fitness_formula": "0.6*macro_f1 + 0.3*attack_recall - 0.1*(features_count/28)",
            "fitness_weights": dict(params["weights"]),
            "best_fitness": float(best_fitness),
            "selected_features_count": len(selected_indices),
            "selected_indices": selected_indices,
            "selected_features": selected_features,
            "feature_mask_path": next((item for item in artifacts if item.endswith("feature_mask.json")), None),
        },
        "dataset": {
            "input_dim_original": 28,
            "input_dim_selected": len(selected_indices),
            "train_npz": config["inputs"]["train_npz"],
            "val_npz": config["inputs"]["val_npz"],
            "test_npz": config["inputs"]["test_npz"],
            "test_used_for_selection": False,
        },
        "validation": best_metrics,
        "test": {"used_for_selection": False, "evaluated": False},
        "comparison": {},
        "artifacts": artifacts,
        "figures": figures,
        "criteria": criteria,
        "warnings": warnings or [],
        "errors": errors or [],
    }
    return summary


def write_latest_pointers(
    *,
    config: dict[str, Any],
    run_id: str,
    run_dir: str | Path,
    qga_dir: str | Path,
    summary: dict[str, Any],
) -> None:
    latest = {
        "run_id": run_id,
        "run_dir": relative_to_repo(run_dir, config),
        "run_summary": relative_to_repo(Path(run_dir) / "artifacts" / "run_summary.json", config),
    }
    write_json(Path(qga_dir) / "latest_run.json", latest)
    write_json(Path(qga_dir) / "latest_run_summary.json", summary)


def build_markdown_report(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected = "\n".join(f"- {name}" for name in summary["qga"]["selected_features"])
    content = f"""# P8 — QGA Feature Selection Report

## 1. Objective

Select a reduced L1 feature subset with a quantum-inspired genetic algorithm.

## 2. Run

- Run ID: `{summary['run_id']}`
- Mode: `{summary['mode']}`
- Accepted: `{summary['accepted']}`

## 3. Fitness

`{summary['qga']['fitness_formula']}`

## 4. Selected Features

Count: {summary['qga']['selected_features_count']} / 28

{selected}

## 5. Validation Metrics

- Macro-F1: {summary['validation'].get('macro_f1')}
- Attack recall: {summary['validation'].get('recall_attack')}
- FPR: {summary['validation'].get('FPR')}
- Fitness: {summary['qga']['best_fitness']}

## 6. Test Holdout

The global test holdout was not used for mask selection.

## 7. Artifacts

{chr(10).join(f'- `{artifact}`' for artifact in summary['artifacts'])}

## 8. Figures

{chr(10).join(f'- `{figure}`' for figure in summary['figures'])}

## 9. Conclusion P8

P8 QGA standalone is code-ready when `accepted=true`; FedAvg/HeteroFL full validation remains a user-triggered step.
"""
    out.write_text(content, encoding="utf-8")
