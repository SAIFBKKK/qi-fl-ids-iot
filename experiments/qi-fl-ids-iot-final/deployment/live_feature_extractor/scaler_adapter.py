from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any


def default_scaler_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "artifacts"
        / "scalers"
        / "l1_binary_robust_scaler.pkl"
    )


def scaler_exists(path: str | Path | None = None) -> bool:
    scaler_path = Path(path) if path else default_scaler_path()
    return scaler_path.exists()


def load_scaler(path: str | Path | None = None) -> Any | None:
    scaler_path = Path(path) if path else default_scaler_path()
    if not scaler_path.exists():
        return None
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the L1 robust scaler") from exc
    return joblib.load(scaler_path)


def ensure_scalable_rows(rows: list[list[float | None]]) -> None:
    for row_index, row in enumerate(rows, start=1):
        if len(row) != 28:
            raise ValueError(f"scaler expects 28 features, row {row_index} has {len(row)}")
        for feature_index, value in enumerate(row):
            if value is None:
                raise ValueError(f"cannot scale row {row_index}: feature index {feature_index} is unsupported")
            if not math.isfinite(float(value)):
                raise ValueError(f"cannot scale row {row_index}: feature index {feature_index} is not finite")


def scale_original_28_rows(rows: list[list[float | None]], path: str | Path | None = None) -> tuple[list[list[float]], bool]:
    scaler = load_scaler(path)
    if scaler is None:
        return [], False
    ensure_scalable_rows(rows)
    matrix = [[float(value) for value in row] for row in rows]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        transformed = scaler.transform(matrix)
    return [[float(value) for value in row] for row in transformed], True
