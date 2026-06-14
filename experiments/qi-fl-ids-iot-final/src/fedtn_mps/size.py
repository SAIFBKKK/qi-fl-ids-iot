"""Size and parameter accounting for P11."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_size_bytes(model: nn.Module, bytes_per_parameter: int = 4) -> int:
    return int(count_parameters(model) * int(bytes_per_parameter))


def compression_ratio(compressed_bytes: int, dense_bytes: int) -> float:
    return float(compressed_bytes / max(int(dense_bytes), 1))


def bandwidth_bytes(model_bytes: int, clients: int, rounds: int) -> int:
    return int(2 * int(clients) * int(model_bytes) * int(rounds))
