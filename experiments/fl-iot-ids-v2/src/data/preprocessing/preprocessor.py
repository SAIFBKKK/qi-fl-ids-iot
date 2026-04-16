from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.common.logger import get_logger


logger = get_logger("preprocessor")


class LocalNodePreprocessor:
    """
    FL-oriented local preprocessor.

    Designed for scenario/node partitions such as:
        data/raw/<scenario>/<node>/train.csv

    Responsibilities:
    - load one node CSV
    - detect label column
    - infer feature columns automatically
    - encode labels using label_id directly when available
    - fallback to label_mapping.json when textual labels are used
    - fit a LOCAL RobustScaler on this node data
    - save output as NPZ for Flower/PyTorch pipeline
    """

    LABEL_CANDIDATES = ["label_id", "label", "Label"]

    def __init__(self, artifacts_dir: str | Path | None = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self.label_mapping: Dict[str, int] | None = None
        self.scaler: RobustScaler | None = None

    def load_artifacts(self) -> None:
        """
        Optional artifact loading.
        Only label_mapping.json is used if available.
        """
        if self.artifacts_dir is None:
            logger.info("No artifacts directory provided. Proceeding without optional artifacts.")
            return

        label_map_json = self.artifacts_dir / "label_mapping.json"
        if label_map_json.exists():
            with label_map_json.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                raise TypeError("label_mapping.json must contain a JSON object")

            self.label_mapping = {str(k): int(v) for k, v in raw.items()}
            logger.info("Loaded label_mapping.json with %d labels", len(self.label_mapping))
        else:
            logger.info("No label_mapping.json found in %s", self.artifacts_dir)

    @classmethod
    def detect_label_column(cls, df: pd.DataFrame) -> str:
        for col in cls.LABEL_CANDIDATES:
            if col in df.columns:
                return col

        raise ValueError(
            f"No label column found. Expected one of {cls.LABEL_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )

    @classmethod
    def infer_feature_columns(cls, df: pd.DataFrame) -> List[str]:
        feature_cols = [c for c in df.columns if c not in cls.LABEL_CANDIDATES]
        if not feature_cols:
            raise ValueError("No feature columns found after excluding label columns.")
        return feature_cols

    def encode_labels(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        """
        Priority:
        1) use label_id directly if present
        2) otherwise map textual labels using label_mapping.json
        """
        if label_col == "label_id":
            y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int64)
            return y

        if self.label_mapping is None:
            raise ValueError(
                f"Label column '{label_col}' is textual but label_mapping.json is not available."
            )

        y_raw = df[label_col].astype(str)
        unknown_labels = sorted(set(y_raw) - set(self.label_mapping.keys()))
        if unknown_labels:
            raise ValueError(
                f"Unknown textual labels found ({len(unknown_labels)}): {unknown_labels[:20]}"
            )

        y = y_raw.map(self.label_mapping).to_numpy(dtype=np.int64)
        return y

    def fit_local_scaler(self, x_df: pd.DataFrame) -> np.ndarray:
        self.scaler = RobustScaler()
        x_scaled = self.scaler.fit_transform(x_df)
        return np.asarray(x_scaled, dtype=np.float32)

    def transform_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform one node dataframe into:
        - X_scaled: np.ndarray [n_samples, n_features]
        - y_encoded: np.ndarray [n_samples]
        - feature_names: ordered feature list
        """
        label_col = self.detect_label_column(df)
        logger.info("Detected label column: %s", label_col)

        feature_cols = self.infer_feature_columns(df)
        logger.info("Detected %d feature columns", len(feature_cols))

        x_df = df[feature_cols].copy()

        # Ensure all features are numeric
        for col in feature_cols:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

        nan_count = int(x_df.isna().sum().sum())
        if nan_count > 0:
            raise ValueError(
                f"Found {nan_count} NaN values in feature matrix after numeric conversion."
            )

        y_encoded = self.encode_labels(df, label_col)

        logger.info("Fitting local RobustScaler...")
        x_scaled = self.fit_local_scaler(x_df)

        return x_scaled, y_encoded, feature_cols

    def process_csv(self, input_csv: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        logger.info("Loading raw CSV: %s", input_csv)
        df = pd.read_csv(input_csv)
        logger.info("Loaded raw dataframe with shape=%s", df.shape)

        return self.transform_dataframe(df)

    def save_npz(
        self,
        output_path: str | Path,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            feature_names=np.array(feature_names, dtype=object),
        )

        logger.info(
            "Saved preprocessed dataset -> %s | X shape=%s | y shape=%s",
            output_path,
            X.shape,
            y.shape,
        )
        return output_path