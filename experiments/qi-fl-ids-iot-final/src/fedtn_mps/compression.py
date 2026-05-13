"""Compression accounting for FedTN/MPS dry-runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import build_compressed_model, build_dense_model
from .size import bandwidth_bytes, compression_ratio, count_parameters, model_size_bytes


@dataclass(frozen=True)
class CompressionResult:
    rank: int
    dense_num_parameters: int
    compressed_num_parameters: int
    dense_model_size_bytes: int
    compressed_model_size_bytes: int
    compression_ratio: float
    parameter_reduction_ratio: float
    dense_bandwidth_total_bytes: int
    compressed_bandwidth_total_bytes: int
    bandwidth_reduction_ratio: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "dense_num_parameters": self.dense_num_parameters,
            "compressed_num_parameters": self.compressed_num_parameters,
            "dense_model_size_bytes": self.dense_model_size_bytes,
            "compressed_model_size_bytes": self.compressed_model_size_bytes,
            "compression_ratio": self.compression_ratio,
            "parameter_reduction_ratio": self.parameter_reduction_ratio,
            "dense_bandwidth_total_bytes": self.dense_bandwidth_total_bytes,
            "compressed_bandwidth_total_bytes": self.compressed_bandwidth_total_bytes,
            "bandwidth_reduction_ratio": self.bandwidth_reduction_ratio,
            "warnings": self.warnings,
        }


def estimate_low_rank_compression(config: dict[str, Any], *, rank: int) -> CompressionResult:
    dense = build_dense_model(config)
    compressed = build_compressed_model(config, rank=rank)
    dense_params = count_parameters(dense)
    compressed_params = count_parameters(compressed)
    dense_bytes = model_size_bytes(dense)
    compressed_bytes = model_size_bytes(compressed)
    rounds = int(config.get("evaluation", {}).get("rounds", 30))
    clients = int(config.get("evaluation", {}).get("clients", 3))
    dense_bandwidth = bandwidth_bytes(dense_bytes, clients, rounds)
    compressed_bandwidth = bandwidth_bytes(compressed_bytes, clients, rounds)
    warnings: list[str] = []
    if compressed_params >= dense_params:
        warnings.append(f"rank_{rank}_is_not_compressive_for_this_small_model")
    return CompressionResult(
        rank=int(rank),
        dense_num_parameters=dense_params,
        compressed_num_parameters=compressed_params,
        dense_model_size_bytes=dense_bytes,
        compressed_model_size_bytes=compressed_bytes,
        compression_ratio=compression_ratio(compressed_bytes, dense_bytes),
        parameter_reduction_ratio=1.0 - (compressed_params / max(dense_params, 1)),
        dense_bandwidth_total_bytes=dense_bandwidth,
        compressed_bandwidth_total_bytes=compressed_bandwidth,
        bandwidth_reduction_ratio=1.0 - (compressed_bandwidth / max(dense_bandwidth, 1)),
        warnings=warnings,
    )
