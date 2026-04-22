from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR, INPUT_DIM


logger = get_logger("preprocess_node_data")

LABEL_COL = "label_id"

# Scaler search order — first match wins.  Per-node fitting is intentionally
# removed: local scalers break feature-scale consistency across FL clients.
_GLOBAL_SCALER_CANDIDATES = [
    "scaler_standard_global.pkl",  # preferred: StandardScaler fitted on full dataset
    "scaler_global.pkl",           # alternative name used by fit_global_scaler
    "scaler_robust_global.pkl",    # legacy global RobustScaler
    "scaler_robust.pkl",           # very old legacy (baseline artefact)
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one node's local dataset using a globally fitted scaler."
    )
    parser.add_argument("--node-id", type=str, required=True, help="e.g. node1")
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-npz", type=str, default=None)
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Artifacts directory containing the global scaler and feature_names.",
    )
    return parser


def load_global_scaler(artifacts_dir: Path):
    """Load the first available global scaler from the candidate list.

    Returns (scaler, filename) so callers know which artifact was used.
    Raises FileNotFoundError with an actionable message when none is found.
    """
    for name in _GLOBAL_SCALER_CANDIDATES:
        path = artifacts_dir / name
        if path.exists():
            logger.info("Loading global scaler: %s", path)
            with path.open("rb") as f:
                return pickle.load(f), name
    raise FileNotFoundError(
        f"No global scaler found in {artifacts_dir}.\n"
        f"Searched: {_GLOBAL_SCALER_CANDIDATES}\n"
        "Run  python -m src.scripts.fit_global_scaler  first."
    )


# Keep the old private name as an alias so any existing internal callers keep working.
_load_global_scaler = load_global_scaler


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    node_id = args.node_id
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else DATA_DIR / "raw" / node_id / "train.csv"
    )
    output_npz = (
        Path(args.output_npz)
        if args.output_npz
        else DATA_DIR / "processed" / node_id / "train_preprocessed.npz"
    )
    artifacts_dir = Path(args.artifacts_dir)

    logger.info("Preprocessing node: %s", node_id)
    logger.info("Input CSV:   %s", input_csv)
    logger.info("Output NPZ:  %s", output_npz)
    logger.info("Artifacts:   %s", artifacts_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    logger.info("Loaded dataframe shape=%s", df.shape)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COL}' not found. Available columns: {list(df.columns)}"
        )

    y = df[LABEL_COL].to_numpy(dtype=np.int64)

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    if len(feature_cols) != INPUT_DIM:
        logger.warning(
            "Expected %d feature columns but found %d. Proceeding with %d.",
            INPUT_DIM, len(feature_cols), len(feature_cols),
        )

    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    logger.info("Raw features shape: %s  mean=%.4g  std=%.4g", X_raw.shape, X_raw.mean(), X_raw.std())

    scaler, scaler_name = _load_global_scaler(artifacts_dir)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Verification — flag if scaling looks wrong
    node_mean = float(X_scaled.mean())
    node_std = float(X_scaled.std())
    logger.info("=== Post-scaling verification (%s) ===", node_id)
    logger.info("scaler used : %s", scaler_name)
    logger.info("mean=%.6f  std=%.6f  (expected: mean≈0, std≈1)", node_mean, node_std)
    if abs(node_mean) > 0.1:
        logger.warning("Node mean %.4f is far from 0 — verify the global scaler is correct", node_mean)
    if abs(node_std - 1.0) > 0.1:
        logger.warning("Node std %.4f is far from 1 — verify the global scaler is correct", node_std)

    # Save NPZ
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X_scaled,
        y=y,
        feature_names=np.array(feature_cols, dtype=object),
    )
    logger.info(
        "Saved NPZ -> %s | X=%s | y=%s",
        output_npz, X_scaled.shape, y.shape,
    )
    logger.info("Preprocessing completed for %s.", node_id)


if __name__ == "__main__":
    main()
