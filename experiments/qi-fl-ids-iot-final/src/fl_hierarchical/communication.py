"""Communication accounting for P6 hierarchical Flower runs."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def model_size_bytes(state_dict_or_params: Mapping[str, torch.Tensor] | list[np.ndarray]) -> int:
    """Compute tensor payload size in bytes without assuming a network codec."""

    if isinstance(state_dict_or_params, list):
        return int(sum(np.asarray(array).nbytes for array in state_dict_or_params))
    return int(sum(tensor.detach().cpu().numpy().nbytes for tensor in state_dict_or_params.values()))


def round_bandwidth(
    *,
    model_size_bytes_value: int,
    num_clients: int,
    previous_cumulative_bytes: int = 0,
) -> dict[str, float | int]:
    """Flower round bandwidth proxy: download + upload model payload per client."""

    download_bytes = int(model_size_bytes_value) * int(num_clients)
    upload_bytes = int(model_size_bytes_value) * int(num_clients)
    total_bytes = download_bytes + upload_bytes
    cumulative_bytes = int(previous_cumulative_bytes) + total_bytes
    return {
        "upload_bytes": upload_bytes,
        "download_bytes": download_bytes,
        "total_bytes": total_bytes,
        "cumulative_bytes": cumulative_bytes,
        "total_mb": total_bytes / (1024**2),
        "cumulative_mb": cumulative_bytes / (1024**2),
    }


def human_readable_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"
