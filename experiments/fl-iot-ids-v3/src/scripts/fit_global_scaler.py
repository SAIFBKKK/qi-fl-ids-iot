from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common.logger import get_logger
from src.common.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DATASET_CSV,
    DATASET_PARQUET,
)


logger = get_logger("fit_global_scaler")

LABEL_COL = "label_id"
SCALER_PATH = ARTIFACTS_DIR / "scaler_standard_global.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"


def load_dataset() -> pd.DataFrame:
    if DATASET_PARQUET.exists():
        logger.info("Loading Parquet: %s", DATASET_PARQUET)
        return pd.read_parquet(DATASET_PARQUET)
    if DATASET_CSV.exists():
        logger.info("Parquet not found, loading CSV: %s", DATASET_CSV)
        return pd.read_csv(DATASET_CSV)
    raise FileNotFoundError(
        f"Dataset not found. Tried:\n  {DATASET_PARQUET}\n  {DATASET_CSV}"
    )


def main() -> None:
    df = load_dataset()
    logger.info("Dataset shape=%s", df.shape)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COL}' not found. Available: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    logger.info("Feature columns: %d", len(feature_cols))

    X = df[feature_cols].to_numpy(dtype=np.float64)

    logger.info("Fitting StandardScaler on %d samples...", X.shape[0])
    scaler = StandardScaler()
    scaler.fit(X)

    logger.info("mean  range: [%.4f, %.4f]", scaler.mean_.min(), scaler.mean_.max())
    logger.info("scale range: [%.4f, %.4f]", scaler.scale_.min(), scaler.scale_.max())

    # Verify on a sample to avoid OOM on large datasets
    sample_size = min(500_000, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_check = scaler.transform(X[idx]).astype(np.float32)

    sample_mean = float(X_check.mean())
    sample_std = float(X_check.std())
    logger.info("=== Verification (sample n=%d) ===", sample_size)
    logger.info("mean=%.6f  std=%.6f  (expected: mean≈0, std≈1)", sample_mean, sample_std)

    if abs(sample_mean) > 0.05:
        logger.warning("Post-scale mean %.4f is far from 0 — check for constant/NaN features", sample_mean)
    if abs(sample_std - 1.0) > 0.05:
        logger.warning("Post-scale std %.4f is far from 1 — check for zero-variance features", sample_std)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)
    logger.info("Global scaler saved -> %s", SCALER_PATH)

    with FEATURE_NAMES_PATH.open("wb") as f:
        pickle.dump(feature_cols, f)
    logger.info("Feature names saved -> %s  (%d features)", FEATURE_NAMES_PATH, len(feature_cols))

    logger.info("Next steps:")
    logger.info("  1. python -m src.scripts.prepare_partitions")
    logger.info("  2. python -m src.scripts.preprocess_node_data --node-id node1")
    logger.info("  3. python -m src.scripts.preprocess_node_data --node-id node2")
    logger.info("  4. python -m src.scripts.preprocess_node_data --node-id node3")
    logger.info("  5. python -m src.scripts.generate_weights")


if __name__ == "__main__":
    main()
