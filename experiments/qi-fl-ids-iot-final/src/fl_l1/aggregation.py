"""Pure FedAvg aggregation for P5."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated model state and client weights."""

    state_dict: OrderedDict[str, torch.Tensor]
    weights: dict[str, float]
    total_examples: int


def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> list[np.ndarray]:
    """Convert a model state_dict to numpy arrays in deterministic order."""

    return [tensor.detach().cpu().numpy().copy() for tensor in state_dict.values()]


def numpy_to_state_dict(
    arrays: list[np.ndarray],
    template_state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert arrays back to a state_dict matching a template."""

    if len(arrays) != len(template_state_dict):
        raise ValueError("array count does not match template state_dict")
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for (key, template_tensor), array in zip(template_state_dict.items(), arrays):
        if tuple(array.shape) != tuple(template_tensor.shape):
            raise ValueError(
                f"shape mismatch for {key}: expected {tuple(template_tensor.shape)}, got {tuple(array.shape)}"
            )
        converted[key] = torch.as_tensor(array, dtype=template_tensor.dtype)
    return converted


def _check_shapes(client_states: list[Mapping[str, torch.Tensor]]) -> None:
    if not client_states:
        raise ValueError("FedAvg requires at least one client state")
    reference = client_states[0]
    reference_keys = list(reference.keys())
    reference_shapes = {key: tuple(tensor.shape) for key, tensor in reference.items()}
    for idx, state in enumerate(client_states[1:], start=2):
        if list(state.keys()) != reference_keys:
            raise ValueError(f"client {idx} state_dict keys differ from client 1")
        for key, tensor in state.items():
            if tuple(tensor.shape) != reference_shapes[key]:
                raise ValueError(
                    f"client {idx} shape mismatch for {key}: "
                    f"expected {reference_shapes[key]}, got {tuple(tensor.shape)}"
                )


def fedavg_state_dicts(
    client_states: list[Mapping[str, torch.Tensor]],
    num_examples: list[int],
    *,
    client_ids: list[str] | None = None,
) -> AggregationResult:
    """Compute FedAvg: w_global = sum(n_k / n_total * w_k)."""

    if len(client_states) != len(num_examples):
        raise ValueError("client_states and num_examples length mismatch")
    if any(int(n) <= 0 for n in num_examples):
        raise ValueError("all clients must contribute positive num_examples")
    _check_shapes(client_states)

    total_examples = int(sum(int(n) for n in num_examples))
    ids = client_ids or [f"client_{idx + 1}" for idx in range(len(client_states))]
    weights = {
        client_id: float(int(n) / total_examples)
        for client_id, n in zip(ids, num_examples)
    }

    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in client_states[0].keys():
        value = None
        for state, n in zip(client_states, num_examples):
            term = state[key].detach().cpu().to(torch.float64) * (int(n) / total_examples)
            value = term if value is None else value + term
        assert value is not None
        aggregated[key] = value.to(dtype=client_states[0][key].dtype)

    return AggregationResult(
        state_dict=aggregated,
        weights=weights,
        total_examples=total_examples,
    )
