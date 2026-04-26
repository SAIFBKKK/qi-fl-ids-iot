from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR, NUM_CLASSES


logger = get_logger("generate_weights")


def manifest_path_for_scenario(scenario: str):
    return DATA_DIR / "splits" / f"{scenario}_manifest.json"


def output_path_for_scenario(scenario: str):
    return ARTIFACTS_DIR / f"class_weights_{scenario}.pkl"


def load_global_counts(scenario: str, split: str = "train") -> dict[int, int]:
    """Sum per-class counts across all nodes from the scenario train manifest."""
    manifest_path = manifest_path_for_scenario(scenario)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for scenario={scenario!r}: {manifest_path}. "
            f"Run: python -m src.scripts.generate_scenarios --scenario {scenario}"
        )

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    if "splits" not in manifest:
        raise ValueError(
            f"Legacy manifest detected for scenario={scenario!r}: {manifest_path}. "
            "This manifest predates the split-aware v3 pipeline. "
            f"Regenerate it with: python -m src.scripts.generate_scenarios --scenario {scenario}"
        )

    try:
        nodes = manifest["splits"][split]["nodes"]
    except KeyError as exc:
        raise KeyError(
            f"Manifest {manifest_path} does not contain splits.{split}.nodes"
        ) from exc

    global_counts: dict[int, int] = {}
    for node_info in nodes.values():
        for cls_str, count in node_info["class_distribution"].items():
            cls = int(cls_str)
            global_counts[cls] = global_counts.get(cls, 0) + count

    return global_counts


def compute_weights(counts: dict[int, int], num_classes: int) -> np.ndarray:
    """
    Inverse-frequency weighting, normalised to mean=1.

      w[c] = total / (num_classes * count[c])
      w    = w / mean(w)

    Rare classes get higher weights; frequent classes get lower weights.
    No manual per-class boosting — let the data drive the balance.
    """
    count_array = np.array([counts.get(c, 1) for c in range(num_classes)], dtype=np.float64)
    total = count_array.sum()

    weights = total / (num_classes * count_array)
    weights = weights / weights.mean()  # normalise so mean weight == 1

    return weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario-specific class weights.")
    parser.add_argument("--scenario", type=str, required=True)
    args = parser.parse_args()

    logger.info("Loading class counts from scenario manifest: %s", args.scenario)
    counts = load_global_counts(args.scenario, split="train")

    missing = [c for c in range(NUM_CLASSES) if c not in counts]
    if missing:
        logger.warning("Classes absent from manifest (defaulting to 1): %s", missing)

    total = sum(counts.values())
    logger.info("Global class counts (%d classes, total=%d):", NUM_CLASSES, total)
    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        logger.info("  class %2d : %8d  (%.4f%%)", c, n, 100.0 * n / total if total else 0)

    weights = compute_weights(counts, NUM_CLASSES)

    logger.info("Class weights (inverse-frequency, mean-normalised):")
    for c, w in enumerate(weights):
        logger.info("  class %2d : %.6f", c, w)

    logger.info(
        "Weight stats: min=%.6f  max=%.6f  mean=%.6f  sum=%.4f",
        weights.min(), weights.max(), weights.mean(), weights.sum(),
    )

    output_path = output_path_for_scenario(args.scenario)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(weights, f)

    logger.info("Saved -> %s", output_path)


if __name__ == "__main__":
    main()
