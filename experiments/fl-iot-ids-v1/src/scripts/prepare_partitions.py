from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.logger import get_logger
from src.common.paths import ROOT_DIR, DATA_DIR
from src.common.utils import set_seed


logger = get_logger("prepare_partitions")


DEFAULT_SEED = 42
DEFAULT_NODE_RATIOS = {
    "node1": 0.50,
    "node2": 0.30,
    "node3": 0.20,
}


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect the target column in a robust way."""
    candidates = ["label", "Label"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No label column found. Expected one of: "
        f"{candidates}. Available columns: {list(df.columns)}"
    )


def validate_ratios(node_ratios: Dict[str, float]) -> None:
    """Ensure partition ratios are valid."""
    total = sum(node_ratios.values())
    if not abs(total - 1.0) < 1e-8:
        raise ValueError(
            f"Node ratios must sum to 1.0, but got {total:.6f}. "
            f"Ratios: {node_ratios}"
        )

    for node_id, ratio in node_ratios.items():
        if ratio <= 0:
            raise ValueError(
                f"Each node ratio must be > 0. Invalid ratio for {node_id}: {ratio}"
            )


def resolve_source_train_csv() -> Path:
    """
    Locate the baseline train CSV.
    Assumes the new repo lives beside experiments/baseline-CIC_IOT_2023.
    """
    baseline_train = (
        ROOT_DIR.parent
        / "baseline-CIC_IOT_2023"
        / "raw"
        / "train"
        / "train.csv"
    )

    if not baseline_train.exists():
        raise FileNotFoundError(
            f"Baseline train CSV not found at: {baseline_train}\n"
            "Update resolve_source_train_csv() if your project layout differs."
        )

    return baseline_train


def ensure_output_dirs(node_ids: List[str]) -> None:
    """Create output folders for raw node partitions and split manifests."""
    for node_id in node_ids:
        node_dir = DATA_DIR / "raw" / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

    (DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)


def save_partition(df: pd.DataFrame, node_id: str) -> Path:
    """Save one node partition."""
    output_path = DATA_DIR / "raw" / node_id / "train.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved %s rows for %s -> %s", len(df), node_id, output_path)
    return output_path


def compute_class_distribution(
    df: pd.DataFrame, label_col: str
) -> Dict[str, int]:
    """Return per-class counts as JSON-serializable dict."""
    counts = df[label_col].value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def split_three_way_stratified(
    df: pd.DataFrame,
    label_col: str,
    node_ratios: Dict[str, float],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """
    Build 3 partitions with approximate stratification and different sizes.

    Strategy:
    1) Split node1 vs rest
    2) Split rest into node2 vs node3
    """
    node_ids = list(node_ratios.keys())
    if len(node_ids) != 3:
        raise ValueError(
            "This V1 implementation currently supports exactly 3 nodes."
        )

    n1, n2, n3 = node_ids
    r1, r2, r3 = node_ratios[n1], node_ratios[n2], node_ratios[n3]

    logger.info(
        "Requested ratios -> %s: %.2f, %s: %.2f, %s: %.2f",
        n1, r1, n2, r2, n3, r3
    )

    df_node1, df_rest = train_test_split(
        df,
        test_size=(1.0 - r1),
        stratify=df[label_col],
        random_state=seed,
        shuffle=True,
    )

    # Relative ratio of node3 inside the remaining pool
    rest_total = r2 + r3
    node3_relative = r3 / rest_total

    df_node2, df_node3 = train_test_split(
        df_rest,
        test_size=node3_relative,
        stratify=df_rest[label_col],
        random_state=seed,
        shuffle=True,
    )

    return {
        n1: df_node1.reset_index(drop=True),
        n2: df_node2.reset_index(drop=True),
        n3: df_node3.reset_index(drop=True),
    }


def build_manifest(
    source_csv: Path,
    full_df: pd.DataFrame,
    label_col: str,
    partitions: Dict[str, pd.DataFrame],
    seed: int,
    node_ratios: Dict[str, float],
) -> Dict:
    """Create metadata manifest for traceability."""
    manifest = {
        "source_train_csv": str(source_csv),
        "seed": seed,
        "label_column": label_col,
        "total_rows": int(len(full_df)),
        "total_columns": int(full_df.shape[1]),
        "node_ratios": node_ratios,
        "global_class_distribution": compute_class_distribution(full_df, label_col),
        "nodes": {},
    }

    for node_id, part_df in partitions.items():
        manifest["nodes"][node_id] = {
            "rows": int(len(part_df)),
            "columns": int(part_df.shape[1]),
            "output_csv": str(DATA_DIR / "raw" / node_id / "train.csv"),
            "class_distribution": compute_class_distribution(part_df, label_col),
        }

    return manifest


def save_manifest(manifest: Dict) -> Path:
    """Save partition manifest to JSON."""
    manifest_path = DATA_DIR / "splits" / "partition_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Partition manifest saved -> %s", manifest_path)
    return manifest_path


def print_summary(manifest: Dict) -> None:
    """Print a concise execution summary."""
    logger.info("=== Partition Summary ===")
    logger.info("Source: %s", manifest["source_train_csv"])
    logger.info("Label column: %s", manifest["label_column"])
    logger.info("Total rows: %s", manifest["total_rows"])

    for node_id, info in manifest["nodes"].items():
        logger.info(
            "%s -> rows=%s, output=%s",
            node_id,
            info["rows"],
            info["output_csv"],
        )


def main() -> None:
    set_seed(DEFAULT_SEED)
    validate_ratios(DEFAULT_NODE_RATIOS)

    logger.info("Starting partition preparation...")

    source_csv = resolve_source_train_csv()
    logger.info("Loading source CSV: %s", source_csv)

    df = pd.read_csv(source_csv)
    logger.info("Loaded dataframe with shape=%s", df.shape)

    label_col = detect_label_column(df)
    logger.info("Detected label column: %s", label_col)

    node_ids = list(DEFAULT_NODE_RATIOS.keys())
    ensure_output_dirs(node_ids)

    partitions = split_three_way_stratified(
        df=df,
        label_col=label_col,
        node_ratios=DEFAULT_NODE_RATIOS,
        seed=DEFAULT_SEED,
    )

    for node_id, part_df in partitions.items():
        save_partition(part_df, node_id)

    manifest = build_manifest(
        source_csv=source_csv,
        full_df=df,
        label_col=label_col,
        partitions=partitions,
        seed=DEFAULT_SEED,
        node_ratios=DEFAULT_NODE_RATIOS,
    )

    save_manifest(manifest)
    print_summary(manifest)

    logger.info("Partition preparation completed successfully.")


if __name__ == "__main__":
    main()