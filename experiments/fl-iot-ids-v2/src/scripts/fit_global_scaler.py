from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR


logger = get_logger("fit_global_scaler")

LABEL_CANDIDATES = {"label_id", "label", "Label"}
METADATA_COLUMNS = {"__row_id"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in LABEL_CANDIDATES | METADATA_COLUMNS]


def fit_global_scaler_for_scenario(scenario: str, artifacts_dir: Path = ARTIFACTS_DIR / "shared") -> Path:
    raw_root = DATA_DIR / "raw" / scenario
    node_paths = sorted(raw_root.glob("node*/train.csv"))
    if not node_paths:
        raise FileNotFoundError(f"No node train CSV files found under {raw_root}")

    frames = []
    feature_cols: list[str] | None = None
    for path in node_paths:
        df = pd.read_csv(path)
        cols = _feature_columns(df)
        if feature_cols is None:
            feature_cols = cols
        elif cols != feature_cols:
            raise ValueError(
                f"Feature order mismatch in {path}. "
                f"Expected first columns {feature_cols[:10]}, got {cols[:10]}"
            )
        frames.append(df[cols])

    assert feature_cols is not None
    train_df = pd.concat(frames, axis=0, ignore_index=True)
    scaler = RobustScaler()
    scaler.fit(train_df)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = artifacts_dir / f"scaler_global_{scenario}.pkl"
    feature_path = artifacts_dir / f"feature_names_{scenario}.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    with feature_path.open("wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Fitted global RobustScaler on %d train rows", len(train_df))
    logger.info("Saved scaler -> %s", scaler_path)
    logger.info("Saved feature names -> %s", feature_path)
    return scaler_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit v2 global train-only scaler.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR / "shared"))
    args = parser.parse_args()

    fit_global_scaler_for_scenario(args.scenario, artifacts_dir=Path(args.artifacts_dir))


if __name__ == "__main__":
    main()

