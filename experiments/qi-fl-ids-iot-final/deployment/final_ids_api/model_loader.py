from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class L1MLPNotAvailable(RuntimeError):
    pass


def bundle_dir() -> Path:
    return Path(os.getenv("DEPLOYMENT_BUNDLE_DIR", Path(__file__).resolve().parents[1] / "l1_final"))


class TorchL1Model:
    def __init__(self, bundle: Path | None = None) -> None:
        self.bundle = bundle or bundle_dir()
        self.selected_model = self._read_json(self.bundle / "selected_model.json")
        self.manifest = self._read_json(self.bundle / "deployment_manifest.json")
        self.model = None
        self.ready = False
        self.error = ""
        self._load()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load(self) -> None:
        checkpoint = self.bundle / "artifacts" / "model.pth"
        if not checkpoint.exists():
            self.error = "model artifact unavailable"
            return
        try:
            import torch
            from torch import nn

            class CentralizedL1MLP(nn.Module):
                def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, dropout: float = 0.2) -> None:
                    super().__init__()
                    layers: list[nn.Module] = []
                    previous_dim = input_dim
                    for hidden_dim in hidden_layers:
                        layers.append(nn.Linear(previous_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        if dropout > 0:
                            layers.append(nn.Dropout(dropout))
                        previous_dim = hidden_dim
                    layers.append(nn.Linear(previous_dim, output_dim))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):  # type: ignore[no-untyped-def]
                    return self.network(x)

            input_dim = int(self.selected_model.get("input_dim", 12))
            hidden_layers = [int(value) for value in self.selected_model.get("hidden_layers", [128, 64])]
            output_dim = int(self.selected_model.get("output_dim", 2))
            model = CentralizedL1MLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, dropout=0.2)
            payload = torch.load(checkpoint, map_location="cpu")
            state = payload
            if isinstance(payload, dict):
                for key in ["model_state_dict", "state_dict", "model"]:
                    if isinstance(payload.get(key), dict):
                        state = payload[key]
                        break
            model.load_state_dict(state, strict=True)
            model.eval()
            self.model = model
            self.ready = True
            self.error = ""
        except Exception as exc:  # noqa: BLE001 - service must expose readiness instead of crashing
            self.ready = False
            self.error = f"model load failed: {exc}"

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        if not self.ready or self.model is None:
            raise L1MLPNotAvailable(self.error or "model artifact unavailable")
        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(batch, dtype=torch.float32)
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()
