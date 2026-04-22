from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.logger import get_logger


logger = get_logger("preprocessor")


class BaselinePreprocessor:
    """
    Preprocessor aligned with the centralized baseline.

    Responsibilities:
    - load baseline artifacts
    - enforce baseline feature order
    - apply baseline scaler
    - encode labels according to baseline mapping
    """

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)

        self.feature_names: List[str] | None = None
        self.scaler: Any = None
        self.label_mapping_raw: Dict[Any, Any] | None = None
        self.label_to_index: Dict[str, int] | None = None

    def load_artifacts(self) -> None:
        """Load baseline artifacts from disk."""
        feature_path = self.artifacts_dir / "feature_names.pkl"
        scaler_path = self.artifacts_dir / "scaler_robust.pkl"
        label_map_path = self.artifacts_dir / "label_mapping_34.pkl"

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing artifact: {feature_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing artifact: {scaler_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Missing artifact: {label_map_path}")

        with feature_path.open("rb") as f:
            self.feature_names = pickle.load(f)

        with scaler_path.open("rb") as f:
            self.scaler = pickle.load(f)

        with label_map_path.open("rb") as f:
            self.label_mapping_raw = pickle.load(f)

        self.label_to_index = self._normalize_label_mapping(self.label_mapping_raw)

        logger.info("Artifacts loaded successfully.")
        logger.info("Number of baseline features: %d", len(self.feature_names))
        logger.info("Number of labels in mapping: %d", len(self.label_to_index))

    @staticmethod
    def detect_label_column(df: pd.DataFrame) -> str:
        """Detect label column name."""
        candidates = ["label", "Label"]
        for col in candidates:
            if col in df.columns:
                return col

        raise ValueError(
            f"No label column found. Expected one of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    @staticmethod
    def _normalize_label_mapping(mapping: Dict[Any, Any]) -> Dict[str, int]:
        """
        Normalize different possible mapping formats into:
            {label_text: class_index}

        Supported examples:
        - {"BenignTraffic": 0, "XSS": 1}
        - {0: "BenignTraffic", 1: "XSS"}
        """
        if not isinstance(mapping, dict):
            raise TypeError(
                f"label_mapping must be a dict, got {type(mapping).__name__}"
            )

        if not mapping:
            raise ValueError("label_mapping is empty")

        sample_key = next(iter(mapping.keys()))
        sample_value = mapping[sample_key]

        # Case 1: already str -> int
        if isinstance(sample_key, str) and isinstance(sample_value, (int, np.integer)):
            return {str(k): int(v) for k, v in mapping.items()}

        # Case 2: int -> str, invert it
        if isinstance(sample_key, (int, np.integer)) and isinstance(sample_value, str):
            return {str(v): int(k) for k, v in mapping.items()}

        # Case 3: nested dict like {'label_to_id': {'label': id, ...}}
        if sample_key == 'label_to_id' and isinstance(sample_value, dict):
            nested = sample_value
            nested_key = next(iter(nested.keys()))
            nested_value = nested[nested_key]
            if isinstance(nested_key, str) and isinstance(nested_value, (int, np.integer)):
                return {str(k): int(v) for k, v in nested.items()}

        raise ValueError(
            "Unsupported label mapping format. "
            f"Example entry: key={sample_key} ({type(sample_key).__name__}), "
            f"value={sample_value} ({type(sample_value).__name__})"
        )

    def validate_feature_columns(self, df: pd.DataFrame) -> None:
        """Check whether all baseline features exist in the dataframe."""
        assert self.feature_names is not None, "Artifacts must be loaded first."

        missing = [col for col in self.feature_names if col not in df.columns]
        extra = [col for col in df.columns if col not in self.feature_names]

        if missing:
            raise ValueError(
                f"Missing {len(missing)} baseline feature(s): {missing[:20]}"
            )

        logger.info("All baseline features are present.")
        logger.info("Extra columns detected outside baseline feature set: %d", len(extra))

    def reorder_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder dataframe columns according to baseline feature_names."""
        assert self.feature_names is not None, "Artifacts must be loaded first."
        return df[self.feature_names].copy()

    def encode_labels(self, y_raw: pd.Series) -> np.ndarray:
        """Convert textual labels to integer class ids."""
        assert self.label_to_index is not None, "Artifacts must be loaded first."

        unknown_labels = sorted(set(y_raw.astype(str)) - set(self.label_to_index.keys()))
        if unknown_labels:
            raise ValueError(
                f"Unknown labels found ({len(unknown_labels)}): {unknown_labels[:20]}"
            )

        y_encoded = y_raw.astype(str).map(self.label_to_index).to_numpy(dtype=np.int64)
        return y_encoded

    def transform_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform a raw dataframe into:
        - X_scaled: np.ndarray [n_samples, n_features]
        - y_encoded: np.ndarray [n_samples]
        - feature_names: ordered feature list
        """
        assert self.feature_names is not None, "Artifacts must be loaded first."
        assert self.scaler is not None, "Artifacts must be loaded first."

        label_col = self.detect_label_column(df)
        logger.info("Detected label column: %s", label_col)

        y_raw = df[label_col].copy()
        X_df = df.drop(columns=[label_col])

        self.validate_feature_columns(X_df)
        X_ordered = self.reorder_features(X_df)

        logger.info("Applying baseline scaler...")
        X_scaled = self.scaler.transform(X_ordered)
        X_scaled = np.asarray(X_scaled, dtype=np.float32)

        logger.info("Encoding labels...")
        y_encoded = self.encode_labels(y_raw)

        return X_scaled, y_encoded, list(self.feature_names)

    def process_csv(self, input_csv: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load a CSV and preprocess it."""
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
        """Save processed arrays into a compressed NPZ file."""
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