from __future__ import annotations

from pathlib import Path


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


def load_scaler(path: str | Path | None = None):
    scaler_path = Path(path) if path else default_scaler_path()
    if not scaler_path.exists():
        return None
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the L1 robust scaler") from exc
    return joblib.load(scaler_path)


def apply_scaler(rows: list[list[float | None]], path: str | Path | None = None) -> tuple[list[list[float | None]], bool]:
    scaler = load_scaler(path)
    if scaler is None:
        return rows, False
    if any(value is None for row in rows for value in row):
        raise ValueError("cannot scale rows containing null unsupported features")
    matrix = [[float(value) for value in row] for row in rows]
    transformed = scaler.transform(matrix)
    return [[float(value) for value in row] for row in transformed], True

