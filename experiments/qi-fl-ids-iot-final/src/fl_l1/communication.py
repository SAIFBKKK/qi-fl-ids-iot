"""Deterministic communication accounting for P5 FedAvg."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def model_size_bytes(state_dict: Mapping[str, torch.Tensor] | list[np.ndarray]) -> int:
    """Compute serialized tensor payload size without networking."""

    if isinstance(state_dict, list):
        return int(sum(np.asarray(array).nbytes for array in state_dict))
    return int(sum(tensor.detach().cpu().numpy().nbytes for tensor in state_dict.values()))


def round_bandwidth(
    *,
    model_size_bytes_value: int,
    num_clients: int,
    previous_cumulative_bytes: int = 0,
) -> dict[str, float | int]:
    """Compute upload/download/total/cumulative bytes for one FedAvg round."""

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
    """Format bytes as B/KB/MB/GB."""

    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"
