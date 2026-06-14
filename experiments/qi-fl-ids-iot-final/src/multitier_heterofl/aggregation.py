"""Slice-weighted aggregation for P7 HeteroFL."""

from __future__ import annotations

from typing import Any

import torch

from multitier_heterofl.slicing import contribution_indices


def expand_subtensor_to_global(
    sub_value: torch.Tensor,
    indices: tuple[slice, ...],
    global_shape: tuple[int, ...],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = device or sub_value.device
    expanded = torch.zeros(global_shape, dtype=torch.float32, device=target_device)
    mask = torch.zeros(global_shape, dtype=torch.float32, device=target_device)
    expanded[indices] = sub_value.detach().float().to(target_device)
    mask[indices] = 1.0
    return expanded, mask


def aggregate_slice_weighted(
    client_updates: list[dict[str, Any]],
    global_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Aggregate only contributed slices, keeping old global values otherwise."""

    if not client_updates:
        return {key: value.detach().clone() for key, value in global_state.items()}, {
            "updated_ratio": 0.0,
            "updated_tensors": {},
        }
    weighted_sum = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in global_state.items()}
    counts = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in global_state.items()}
    client_counts = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in global_state.items()}
    for update in client_updates:
        tier = str(update["tier"])
        state = update["state_dict"]
        n = float(update["num_examples"])
        idx_map = contribution_indices(tier)
        for key, indices in idx_map.items():
            expected = tuple(global_state[key][indices].shape)
            actual = tuple(state[key].shape)
            if actual != expected:
                raise ValueError(f"slice shape mismatch for {key} tier={tier}: expected {expected}, got {actual}")
            expanded, mask = expand_subtensor_to_global(
                state[key],
                indices,
                tuple(global_state[key].shape),
                device=global_state[key].device,
            )
            weighted_sum[key].add_(expanded, alpha=n)
            counts[key].add_(mask, alpha=n)
            client_counts[key] += mask
    new_state: dict[str, torch.Tensor] = {}
    updated_elements = 0
    total_elements = 0
    updated_tensors: dict[str, int] = {}
    for key, old_value in global_state.items():
        has = counts[key] > 0
        safe_counts = torch.where(has, counts[key], torch.ones_like(counts[key]))
        aggregated = weighted_sum[key] / safe_counts
        new_state[key] = torch.where(has, aggregated, old_value.float()).to(dtype=old_value.dtype)
        updated = int(has.sum().item())
        updated_tensors[key] = updated
        updated_elements += updated
        total_elements += int(has.numel())
    return new_state, {
        "updated_ratio": float(updated_elements / max(total_elements, 1)),
        "updated_elements": int(updated_elements),
        "total_elements": int(total_elements),
        "updated_tensors": updated_tensors,
        "contributor_counts_max": {key: int(value.max().item()) for key, value in client_counts.items()},
    }
