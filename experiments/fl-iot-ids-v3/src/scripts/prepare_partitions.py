from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.paths import DATA_DIR, DATASET_CSV, DATASET_PARQUET
from src.common.utils import set_seed


logger = get_logger("prepare_partitions")

DEFAULT_SEED = 42
NUM_NODES = 3
DIRICHLET_ALPHA = 0.5
LABEL_COL = "label_id"
RARE_THRESHOLD = 1000
MIN_SAMPLES_PER_CLASS = 200  # floor guarantee per class per node


def load_dataset() -> pd.DataFrame:
    """Load dataset from Parquet (fast) or CSV (fallback)."""
    if DATASET_PARQUET.exists():
        logger.info("Loading from Parquet: %s", DATASET_PARQUET)
        df = pd.read_parquet(DATASET_PARQUET)
    elif DATASET_CSV.exists():
        logger.info("Parquet not found, falling back to CSV: %s", DATASET_CSV)
        df = pd.read_csv(DATASET_CSV)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Tried:\n  {DATASET_PARQUET}\n  {DATASET_CSV}"
        )
    logger.info("Loaded dataset shape=%s", df.shape)
    return df


def dirichlet_partition(
    df: pd.DataFrame,
    num_nodes: int,
    alpha: float,
    seed: int,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> Dict[str, pd.DataFrame]:
    """
    Non-IID partition via Dirichlet(alpha) with a per-class floor guarantee.

    For each class c:
    1. Draw proportions ~ Dirichlet(alpha).
    2. Apply floor: if a node would receive fewer than min_samples, steal the
       deficit from the richest node — but only when the class has enough total
       samples (n >= min_samples * num_nodes). Globally rare classes are left
       as-is so we never create duplicate samples.
    3. Absorb rounding remainder into the last bucket.
    """
    rng = np.random.default_rng(seed)
    node_indices: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

    classes = sorted(df[LABEL_COL].unique())
    logger.info(
        "Partitioning %d classes across %d nodes (alpha=%.2f, min_samples=%d)...",
        len(classes), num_nodes, alpha, min_samples,
    )

    skipped_floor: List[int] = []

    for cls in classes:
        cls_idx = df.index[df[LABEL_COL] == cls].tolist()
        n = len(cls_idx)
        if n == 0:
            continue

        rng.shuffle(cls_idx)

        proportions = rng.dirichlet([alpha] * num_nodes)
        counts = (proportions * n).astype(int)

        # Floor guarantee — only when the class has enough samples globally
        if n >= min_samples * num_nodes:
            for i in range(num_nodes):
                if counts[i] < min_samples:
                    deficit = min_samples - counts[i]
                    richest = int(np.argmax(counts))
                    if counts[richest] - deficit >= min_samples:
                        counts[richest] -= deficit
                        counts[i] = min_samples
        else:
            skipped_floor.append(int(cls))

        # Absorb rounding remainder; clamp negatives caused by edge cases
        counts[-1] = n - counts[:-1].sum()
        counts = np.maximum(counts, 0)

        pos = 0
        for node_i, count in enumerate(counts):
            node_indices[node_i].extend(cls_idx[pos : pos + count])
            pos += count

    if skipped_floor:
        logger.warning(
            "Floor guarantee skipped for %d globally rare classes (n < %d): %s",
            len(skipped_floor), min_samples * num_nodes, skipped_floor,
        )

    partitions: Dict[str, pd.DataFrame] = {}
    for node_i in range(num_nodes):
        node_id = f"node{node_i + 1}"
        idx = node_indices[node_i]
        part = df.loc[idx].reset_index(drop=True)
        part = part.sample(frac=1, random_state=seed).reset_index(drop=True)
        partitions[node_id] = part
        logger.info("  %s -> %d rows", node_id, len(part))

    return partitions


def ensure_output_dirs(node_ids: List[str]) -> None:
    for node_id in node_ids:
        (DATA_DIR / "raw" / node_id).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)


def compute_node_stats(part_df: pd.DataFrame) -> dict:
    class_counts = part_df[LABEL_COL].value_counts().sort_index()
    dist = {int(k): int(v) for k, v in class_counts.items()}
    vals = list(class_counts.values)
    imbalance_ratio = round(float(max(vals)) / float(min(vals)), 2) if vals else 0.0
    rare_classes = [int(k) for k, v in class_counts.items() if v < RARE_THRESHOLD]
    return {
        "rows": int(len(part_df)),
        "class_distribution": dist,
        "imbalance_ratio": imbalance_ratio,
        "rare_classes": rare_classes,
    }


def main() -> None:
    set_seed(DEFAULT_SEED)
    logger.info("Starting Dirichlet partition preparation (alpha=%.2f)...", DIRICHLET_ALPHA)

    df = load_dataset()

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COL}' not found. Available columns: {list(df.columns)}"
        )

    node_ids = [f"node{i + 1}" for i in range(NUM_NODES)]
    ensure_output_dirs(node_ids)

    partitions = dirichlet_partition(df, NUM_NODES, DIRICHLET_ALPHA, DEFAULT_SEED)

    source_path = DATASET_PARQUET if DATASET_PARQUET.exists() else DATASET_CSV
    manifest: dict = {
        "dataset_source": str(source_path),
        "label_column": LABEL_COL,
        "total_rows": int(len(df)),
        "num_nodes": NUM_NODES,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "seed": DEFAULT_SEED,
        "nodes": {},
    }

    for node_id, part_df in partitions.items():
        out_path = DATA_DIR / "raw" / node_id / "train.csv"
        part_df.to_csv(out_path, index=False)
        logger.info("Saved %s -> %s", node_id, out_path)
        manifest["nodes"][node_id] = {
            "output_csv": str(out_path),
            **compute_node_stats(part_df),
        }

    manifest_path = DATA_DIR / "splits" / "partition_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest saved -> %s", manifest_path)

    logger.info("=== Partition Summary ===")
    for node_id, info in manifest["nodes"].items():
        logger.info(
            "%s | rows=%d | imbalance=%.1fx | rare_classes=%d",
            node_id,
            info["rows"],
            info["imbalance_ratio"],
            len(info["rare_classes"]),
        )

    logger.info("Partition preparation completed successfully.")


if __name__ == "__main__":
    main()
