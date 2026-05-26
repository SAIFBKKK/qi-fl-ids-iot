from __future__ import annotations


class ScalerRuntime:
    def __init__(self, available: bool = False) -> None:
        self.available = available

    def transform_28(self, features: list[float]) -> list[float]:
        if not self.available:
            raise RuntimeError("Scaler runtime is not packaged in P16.1 step 0.")
        if len(features) != 28:
            raise ValueError(f"expected 28 features, got {len(features)}")
        return [float(value) for value in features]

    def describe(self) -> dict[str, object]:
        return {"available": self.available, "mode": "placeholder"}

