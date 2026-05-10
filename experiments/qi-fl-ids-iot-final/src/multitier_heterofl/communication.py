"""Communication accounting for P7 HeteroFL."""

from __future__ import annotations

from multitier_heterofl.supernet import build_tier_model, model_size_bytes


def tier_model_size(*, tier: str, output_dim: int, dropout: float = 0.2) -> int:
    return model_size_bytes(build_tier_model(tier=tier, output_dim=output_dim, dropout=dropout))


def round_bandwidth_by_tier(
    *,
    tier_mapping: dict[str, str],
    output_dim: int,
    dropout: float = 0.2,
) -> dict[str, object]:
    by_client: dict[str, dict[str, int | str]] = {}
    by_tier: dict[str, int] = {"weak": 0, "medium": 0, "powerful": 0}
    total = 0
    for client_id, tier in tier_mapping.items():
        size = tier_model_size(tier=tier, output_dim=output_dim, dropout=dropout)
        bytes_for_client = 2 * size
        by_client[client_id] = {
            "tier": tier,
            "model_size_bytes": size,
            "upload_bytes": size,
            "download_bytes": size,
            "total_bytes": bytes_for_client,
        }
        by_tier[tier] += bytes_for_client
        total += bytes_for_client
    return {"by_client": by_client, "by_tier": by_tier, "total_bytes": total}
