from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


def resolve_num_classes(
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any] | None = None,
) -> int:
    """Resolve the global output space from config and reject local inference."""
    if "num_classes" not in dataset_cfg:
        raise ValueError(
            "Missing required config key dataset.num_classes. "
            "FL clients must use a global class space, not local label maxima."
        )

    num_classes = int(dataset_cfg["num_classes"])
    if num_classes <= 1:
        raise ValueError(f"dataset.num_classes must be > 1, got {num_classes}")

    if model_cfg is not None:
        for key in ("num_classes", "output_dim"):
            if key in model_cfg and int(model_cfg[key]) != num_classes:
                raise ValueError(
                    f"Model/config output mismatch: dataset.num_classes={num_classes} "
                    f"but model.{key}={model_cfg[key]}."
                )

    return num_classes


def validate_model_output_dim(model: nn.Module, expected_num_classes: int) -> None:
    """Validate that the last linear layer matches the global output space."""
    last_linear: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module

    if last_linear is None:
        raise ValueError("Model output validation failed: no Linear layer found.")

    if int(last_linear.out_features) != int(expected_num_classes):
        raise ValueError(
            "Model output validation failed: final layer has "
            f"{last_linear.out_features} outputs but config requires "
            f"{expected_num_classes} classes."
        )

