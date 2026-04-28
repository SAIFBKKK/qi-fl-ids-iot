from __future__ import annotations

from typing import Any


class EdgeInferenceEngine:
    """Placeholder inference engine for the future local gateway IDS model."""

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict the traffic class from mapped features.

        TODO(P7.5): load scaler, feature ordering, label mapping, and model bundle.
        """
        raise NotImplementedError("local inference is not implemented in P7.2")
