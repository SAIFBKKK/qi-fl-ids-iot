"""Select final P8-b QGA L2 mask from Flower short validations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.calibration import rank_masks
from qga_l2.config import load_config, load_json, repo_path, write_json


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _bool(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    reports = repo_path(config, "outputs.reports_dir")
    rows = _read(reports / "p8b_qga_l2_flower_short_validation.csv")
    weights = {key: float(value) for key, value in config["calibration"]["engineering_score"].items()}
    ranking, warnings = rank_masks(rows, weights=weights, require_short_flower=True, min_scenarios=3, return_warnings=True)
    if not ranking:
        raise RuntimeError("No complete P8-b QGA L2 mask has >=3 true-Flower 5-round scenarios")
    best = ranking[0]
    sweep = {row["mask_id"]: row for row in _read(reports / "p8b_qga_l2_profile_sweep_summary.csv")}
    source = repo_path(config) / sweep[best["mask_id"]]["feature_mask_path"]
    payload = load_json(source)
    final_dir = repo_path(config, "outputs.qga_l2_dir") / "final_selected_mask"
    final_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "accepted": True,
        "phase": "P8-b",
        "selected_mask_id": best["mask_id"],
        "profile": best["profile"],
        "seed": best["seed"],
        "features_count": best["features_count"],
        "engineering_score": best["engineering_score"],
        "test_used_for_selection": False,
        "ranking": ranking,
        "warnings": warnings,
    }
    write_json(final_dir / "feature_mask.json", payload)
    write_json(final_dir / "selected_features.json", payload)
    write_json(final_dir / "selection_decision.json", decision)
    write_json(reports / "p8b_qga_l2_mask_selection_summary.json", decision)
    (reports / "p8b_qga_l2_mask_selection_decision.md").write_text(
        "\n".join(
            [
                "# P8-b QGA L2 Mask Selection",
                "",
                f"Selected mask: `{best['mask_id']}`.",
                "",
                f"- Engineering score: {float(best['engineering_score']):.6f}",
                f"- Scenarios used: {best['scenario_count']}",
                f"- Mean Macro-F1: {float(best['mean_macro_f1']):.6f}",
                f"- Mean Macro-Recall: {float(best['mean_macro_recall']):.6f}",
                f"- Mean Macro-FPR: {float(best['mean_macro_fpr']):.6f}",
                "",
                "Only true Flower 5-round validations with `test_sent_to_clients=false` were eligible.",
                "Test global not used for selection.",
                "",
                f"Ignored invalid metric rows: {warnings['ignored_invalid_metric_rows']['count']}",
                f"Ignored incomplete masks: {warnings['ignored_incomplete_masks']['count']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Selected P8-b QGA L2 mask: {best['mask_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
