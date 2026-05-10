"""Prefix-neuron slicing for P7 HeteroFL."""

from __future__ import annotations

from typing import Mapping

import torch

from multitier_heterofl.supernet import MAX_H1, MAX_H2, TIER_DIMS


def contribution_indices(tier: str) -> dict[str, tuple[slice, ...]]:
    if tier not in TIER_DIMS:
        raise ValueError(f"unknown tier: {tier}")
    h1, h2 = TIER_DIMS[tier]
    head_in = h2 if h2 else h1
    indices = {
        "fc1.weight": (slice(0, h1), slice(None)),
        "fc1.bias": (slice(0, h1),),
        "fc3.weight": (slice(None), slice(0, head_in)),
        "fc3.bias": (slice(None),),
    }
    if h2:
        indices["fc2.weight"] = (slice(0, h2), slice(0, h1))
        indices["fc2.bias"] = (slice(0, h2),)
    return indices


def extract_tier_state(global_state: Mapping[str, torch.Tensor], tier: str) -> dict[str, torch.Tensor]:
    indices = contribution_indices(tier)
    sub_state: dict[str, torch.Tensor] = {}
    for key, idx in indices.items():
        if key == "fc3.weight" and tier == "weak":
            sub_state[key] = global_state[key][:, : TIER_DIMS[tier][0]].detach().clone()
        else:
            sub_state[key] = global_state[key][idx].detach().clone()
    return sub_state


def load_tier_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    model.load_state_dict(dict(state), strict=True)
    return model


def slice_coverage_ratio(tier_updates: list[str]) -> float:
    """Return approximate updated tensor-element ratio for logging."""

    total = MAX_H1 * 28 + MAX_H1 + MAX_H2 * MAX_H1 + MAX_H2
    # fc3 depends on output_dim; this ratio is intentionally structural.
    updated = 0
    seen_fc3 = False
    for tier in tier_updates:
        h1, h2 = TIER_DIMS[tier]
        updated = max(updated, h1 * 28 + h1)
        if h2:
            updated = max(updated, h1 * 28 + h1 + h2 * h1 + h2)
        seen_fc3 = True
    if seen_fc3:
        updated += MAX_H2
        total += MAX_H2
    return float(min(1.0, updated / max(total, 1)))
