"""Select the final QGA mask from P8.1.5 calibration results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.calibration import rank_masks_from_short_validation
from qga.config import load_config, load_json, repo_path, write_json


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _find_sweep_row(config: dict, mask_id: str) -> dict:
    rows = _read_csv(repo_path(config, "outputs.reports_dir") / "p8_qga_profile_sweep_summary.csv")
    for row in rows:
        if row["mask_id"] == mask_id:
            return row
    raise RuntimeError(f"mask_id not found in sweep summary: {mask_id}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _filter_validation_rows(rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    valid_rows: list[dict] = []
    ignored_stale: list[dict[str, Any]] = []
    ignored_runtime: list[dict[str, Any]] = []

    for row in rows:
        rounds = _as_int(row.get("rounds"), default=-1)
        true_flower = _as_bool(row.get("true_flower_runtime"), default=False)
        test_sent_to_clients = _as_bool(row.get("test_sent_to_clients"), default=False)
        reason: list[str] = []
        if rounds != 5:
            reason.append(f"rounds={rounds}")
        if not true_flower:
            reason.append("true_flower_runtime=false")
        if test_sent_to_clients:
            reason.append("test_sent_to_clients=true")
        if reason:
            ignored = {
                "mask_id": row.get("mask_id"),
                "profile": row.get("profile"),
                "seed": row.get("seed"),
                "scenario": row.get("scenario"),
                "rounds": rounds,
                "run_id": row.get("run_id"),
                "reason": ", ".join(reason),
            }
            if rounds != 5:
                ignored_stale.append(ignored)
            else:
                ignored_runtime.append(ignored)
            continue
        valid_rows.append(row)

    warnings = {
        "ignored_stale_short_runs": {
            "count": len(ignored_stale),
            "runs": ignored_stale,
        },
        "ignored_invalid_runtime_rows": {
            "count": len(ignored_runtime),
            "runs": ignored_runtime,
        },
    }
    return valid_rows, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = repo_path(config, "outputs.reports_dir")
    validation_rows = _read_csv(reports_dir / "p8_qga_flower_short_validation.csv")
    filtered_rows, warnings = _filter_validation_rows(validation_rows)
    weights = {key: float(value) for key, value in config["qga_calibration"]["engineering_score"].items()}
    raw_ranking = rank_masks_from_short_validation(filtered_rows, score_weights=weights)
    incomplete_masks = [row for row in raw_ranking if int(row.get("scenario_count", 0)) < 3]
    ranking = [row for row in raw_ranking if int(row.get("scenario_count", 0)) >= 3]
    warnings["ignored_incomplete_masks"] = {
        "count": len(incomplete_masks),
        "minimum_required_scenarios": 3,
        "masks": incomplete_masks,
    }
    if not ranking:
        raise RuntimeError("no complete 5-round true-Flower validation masks available for mask selection")
    best = ranking[0]
    sweep_row = _find_sweep_row(config, best["mask_id"])
    mask_path = repo_path(config, None) / sweep_row["feature_mask_path"]
    feature_mask = load_json(mask_path)
    selected_features_path = mask_path.parent / "selected_features.json"
    selected_features = load_json(selected_features_path)

    final_dir = repo_path(config, "outputs.qga_dir") / "final_selected_mask"
    final_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "accepted": True,
        "phase": "P8.1.5",
        "selected_mask_id": best["mask_id"],
        "profile": best["profile"],
        "seed": int(best["seed"]),
        "engineering_score": best["engineering_score"],
        "selection_reason": "Highest engineering score over validation-only short Flower scenarios.",
        "test_used_for_selection": False,
        "true_flower_short_validation": True,
        "filters": {
            "rounds_required": 5,
            "true_flower_runtime_required": True,
            "test_sent_to_clients_required": False,
            "minimum_scenario_count": 3,
        },
        "warnings": warnings,
        "ranking": ranking,
    }
    write_json(final_dir / "feature_mask.json", feature_mask)
    write_json(final_dir / "selected_features.json", selected_features)
    write_json(final_dir / "selection_decision.json", decision)
    write_json(reports_dir / "p8_qga_mask_selection_summary.json", decision)
    md = f"""# P8.1.5 QGA Mask Selection Decision

## Selected Mask

- Mask ID: `{best['mask_id']}`
- Profile: `{best['profile']}`
- Seed: `{best['seed']}`
- Features: `{best['features_count']}`
- Engineering score: `{best['engineering_score']}`

## Rationale

The selected mask maximizes the engineering score over validation-only true-Flower short runs.
The global test holdout was not used for mask selection.

## Filtering

- Required rounds: `5`
- Required runtime: `true_flower_runtime=true`
- Required test handling: `test_sent_to_clients=false`
- Required scenario count per mask: `>= 3`

## Warnings

- Ignored stale short runs: `{warnings['ignored_stale_short_runs']['count']}`
- Ignored incomplete masks: `{warnings['ignored_incomplete_masks']['count']}`

## Output

- `outputs/qga_feature_selection/final_selected_mask/feature_mask.json`
- `outputs/qga_feature_selection/final_selected_mask/selected_features.json`
- `outputs/qga_feature_selection/final_selected_mask/selection_decision.json`
"""
    (reports_dir / "p8_qga_mask_selection_decision.md").write_text(md, encoding="utf-8")
    print(f"Selected QGA mask: {best['mask_id']} | score={best['engineering_score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
