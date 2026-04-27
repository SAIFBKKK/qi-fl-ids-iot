from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from preprocessor import ProcessedFlow

MODEL_VERSION = "baseline_fedavg_normal_classweights"


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    predicted_label_id: int
    confidence: float
    model_version: str = MODEL_VERSION


class TorchMLPInferenceEngine:
    engine_type = "torch_mlp"

    def __init__(self, model_path: str, label_mapping_path: str) -> None:
        self.model_path = Path(model_path)
        self.label_mapping_path = Path(label_mapping_path)
        self.bundle_path = self.label_mapping_path.parent
        self.model_config_path = self.bundle_path / "model_config.json"
        self.id_to_label = self._load_label_mapping()
        self.model = self._load_model()

    def predict(self, flow: ProcessedFlow) -> PredictionResult:
        tensor = self.torch.tensor(flow.scaled_vector, dtype=self.torch.float32)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)
            confidence, label_id = self.torch.max(probabilities, dim=1)

        predicted_label_id = int(label_id.item())
        return PredictionResult(
            predicted_label_id=predicted_label_id,
            predicted_label=self.id_to_label.get(predicted_label_id, f"class_{predicted_label_id}"),
            confidence=round(float(confidence.item()), 6),
        )

    def _load_model(self) -> Any:
        try:
            import torch
            import torch.nn as nn

            self.torch = torch

            with open(self.model_config_path, encoding="utf-8") as f:
                config = json.load(f)

            hidden_dims = tuple(config.get("hidden_dims", [256, 128]))
            input_dim = int(config["input_dim"])
            num_classes = int(config["num_classes"])
            dropout = float(config.get("dropout", 0.2))

            if input_dim != 28:
                raise ValueError(f"model_config input_dim must be 28, found {input_dim}")

            class MLPClassifier(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    h1, h2 = hidden_dims
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

            model = MLPClassifier()
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint)
            model.eval()
            logger.info(
                "torch_model_loaded",
                path=str(self.model_path),
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=list(hidden_dims),
            )
            return model
        except Exception as exc:  # noqa: BLE001 - P2 requires fail-loud startup.
            self._fail_startup(f"Cannot load model from {self.model_path}: {exc}")

    def _load_label_mapping(self) -> dict[int, str]:
        try:
            with open(self.label_mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)

            if "id_to_label" in mapping:
                id_to_label = {int(key): value for key, value in mapping["id_to_label"].items()}
            elif "label_to_id" in mapping:
                id_to_label = {int(value): key for key, value in mapping["label_to_id"].items()}
            else:
                id_to_label = {int(key): value for key, value in mapping.items()}

            if len(id_to_label) != 34:
                raise ValueError(f"label_mapping must contain 34 classes, found {len(id_to_label)}")
            logger.info("label_mapping_loaded", path=str(self.label_mapping_path), classes=len(id_to_label))
            return id_to_label
        except Exception as exc:  # noqa: BLE001
            self._fail_startup(f"Cannot load label mapping from {self.label_mapping_path}: {exc}")

    @staticmethod
    def _fail_startup(message: str) -> None:
        logger.critical("Cannot load model bundle: {}", message)
        logger.critical("Service refuses to start - fix the bundle first")
        sys.exit(1)


class InferenceService:
    def __init__(self, model_path: str, label_mapping_path: str) -> None:
        self.engine = TorchMLPInferenceEngine(model_path=model_path, label_mapping_path=label_mapping_path)

    @property
    def engine_type(self) -> str:
        return self.engine.engine_type

    def predict(self, flow: ProcessedFlow) -> PredictionResult:
        return self.engine.predict(flow)
