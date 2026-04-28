from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_VERSION = "baseline_fedavg_normal_classweights"


def severity_for_prediction(confidence: float, predicted_label: str) -> str:
    if predicted_label == "BenignTraffic":
        return "none"
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


class EdgeInferenceEngine:
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        model_config_path: str | None = None,
        threshold: float = 0.5,
    ):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.model_path = Path(model_path)
        self.label_mapping_path = Path(label_mapping_path)
        self.model_config_path = (
            Path(model_config_path)
            if model_config_path
            else self.label_mapping_path.parent / "model_config.json"
        )
        self.threshold = float(threshold)

        self.model_config = self._load_model_config()
        self.id_to_label = self._load_label_mapping()
        self.input_dim = int(self.model_config["input_dim"])
        self.num_classes = int(self.model_config["num_classes"])

        if self.input_dim != 28:
            raise ValueError(f"model_config input_dim must be 28, found {self.input_dim}")
        if self.num_classes != len(self.id_to_label):
            raise ValueError(
                f"model_config num_classes={self.num_classes} does not match label count={len(self.id_to_label)}"
            )

        self.hidden_dims = tuple(self.model_config.get("hidden_dims", [256, 128]))
        if len(self.hidden_dims) != 2:
            raise ValueError("model_config hidden_dims must contain exactly two hidden layers")
        self.dropout = float(self.model_config.get("dropout", 0.2))
        self.model = self._load_model()

    def predict(self, scaled_features: Any) -> dict[str, Any]:
        tensor = self.torch.as_tensor(scaled_features, dtype=self.torch.float32)
        if tuple(tensor.shape) != (1, self.input_dim):
            raise ValueError(f"scaled_features must have shape (1, {self.input_dim})")

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)
            confidence_tensor, label_id_tensor = self.torch.max(probabilities, dim=1)

        predicted_label_id = int(label_id_tensor.item())
        predicted_label = self.id_to_label.get(predicted_label_id, f"class_{predicted_label_id}")
        confidence = round(float(confidence_tensor.item()), 6)
        is_alert = confidence >= self.threshold and predicted_label != "BenignTraffic"
        severity = severity_for_prediction(confidence, predicted_label) if is_alert else "none"

        return {
            "predicted_label_id": predicted_label_id,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_alert": is_alert,
            "severity": severity,
            "threshold": self.threshold,
            "model_version": MODEL_VERSION,
        }

    def _load_model_config(self) -> dict[str, Any]:
        try:
            with open(self.model_config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load model_config.json from {self.model_config_path}: {exc}") from exc

    def _load_label_mapping(self) -> dict[int, str]:
        try:
            with open(self.label_mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load label_mapping.json from {self.label_mapping_path}: {exc}") from exc

        if "id_to_label" in mapping:
            id_to_label = {int(key): value for key, value in mapping["id_to_label"].items()}
        elif "label_to_id" in mapping:
            id_to_label = {int(value): key for key, value in mapping["label_to_id"].items()}
        else:
            id_to_label = {int(key): value for key, value in mapping.items()}

        if not id_to_label:
            raise ValueError("label mapping is empty")
        return id_to_label

    def _load_model(self) -> Any:
        h1, h2 = self.hidden_dims
        nn = self.nn

        class MLPClassifier(nn.Module):
            def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, h1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h2, num_classes),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        model = MLPClassifier(self.input_dim, self.num_classes, self.dropout)
        checkpoint = self._load_checkpoint()
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _load_checkpoint(self) -> Any:
        try:
            return self.torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(self.model_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load model checkpoint from {self.model_path}: {exc}") from exc
