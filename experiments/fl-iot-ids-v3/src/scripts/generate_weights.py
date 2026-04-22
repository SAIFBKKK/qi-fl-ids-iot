from __future__ import annotations

import json
import pickle

import numpy as np

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR, NUM_CLASSES


logger = get_logger("generate_weights")

MANIFEST_PATH = DATA_DIR / "splits" / "partition_manifest.json"
OUTPUT_PATH = ARTIFACTS_DIR / "class_weights_34.pkl"


def load_global_counts() -> dict[int, int]:
    """Sum per-class counts across all nodes from the partition manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)

    global_counts: dict[int, int] = {}
    for node_info in manifest["nodes"].values():
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
    logger.info("Loading global class counts from manifest...")
    counts = load_global_counts()

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(weights, f)

    logger.info("Saved -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
