from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

_logger = logging.getLogger(__name__)

_MAX_H1 = 256
_MAX_H2 = 128
_SUPPORTED_WIDTHS = frozenset({0.25, 0.5, 1.0})


def get_contribution_indices(
    width: float,
    max_h1: int = _MAX_H1,
    max_h2: int = _MAX_H2,
) -> dict[str, tuple[slice, ...]]:
    """
    Per-key slices indicating which global SuperNet positions this width trains.

    Static HeteroFL: smaller tiers train the first h1/h2 rows/columns only
    (nested subsets). Input and output dimensions are always fully covered.
    """
    width = float(width)
    if width not in _SUPPORTED_WIDTHS:
        raise ValueError(f"width must be one of {sorted(_SUPPORTED_WIDTHS)}, got {width!r}")
    h1 = int(max_h1 * width)
    h2 = int(max_h2 * width)
    return {
        "fc1.weight": (slice(0, h1), slice(None)),      # [0:h1, :] — first h1 neurons
        "fc1.bias":   (slice(0, h1),),                  # [0:h1]
        "fc2.weight": (slice(0, h2), slice(0, h1)),     # [0:h2, 0:h1]
        "fc2.bias":   (slice(0, h2),),                  # [0:h2]
        "fc3.weight": (slice(None), slice(0, h2)),      # [:, 0:h2] — output fixed
        "fc3.bias":   (slice(None),),                   # [:] — always full
    }


def expand_subtensor_to_global(
    sub_value: Tensor,
    indices: tuple[slice, ...],
    global_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """
    Place sub_value at indices in a zeros tensor of global_shape.

    Returns:
        expanded         : global_shape tensor with sub_value at indices, zeros elsewhere
        contribution_mask: binary tensor, 1 where sub_value was placed
    """
    expanded = torch.zeros(global_shape, dtype=torch.float32)
    mask = torch.zeros(global_shape, dtype=torch.float32)
    expanded[indices] = sub_value.float()
    mask[indices] = 1.0
    return expanded, mask


def aggregate_masked(
    client_updates: list[dict[str, Any]],
    global_state: dict[str, Tensor],
    max_h1: int = _MAX_H1,
    max_h2: int = _MAX_H2,
) -> dict[str, Tensor]:
    """
    Masked weighted average aggregation for heterogeneous-width FL clients.

    Each entry in client_updates must provide:
        state_dict   : dict[str, Tensor]  — sub-model weights (width-specific shape)
        num_examples : int                — number of training samples
        width        : float              — received_width (determines state_dict shape)

    Per-position aggregation:
        new[pos] = Σ(n_i × w_i[pos]) / Σ(n_i)   for contributing clients
        new[pos] = global_state[pos]               if no client contributed to pos

    Never produces NaN or Inf. Extra keys in update dicts (e.g. tier_width) are ignored.
    Returns a full-width state_dict matching global_state keys and shapes.
    """
    if not client_updates:
        return {k: v.clone() for k, v in global_state.items()}

    weighted_sum: dict[str, Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()
    }
    count: dict[str, Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()
    }
    # Tracks number of clients (not weighted) per position — used for logging only
    client_count: dict[str, Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()
    }

    for update in client_updates:
        sub_state: dict[str, Tensor] = update["state_dict"]
        n = float(update["num_examples"])
        width = float(update["width"])
        indices_map = get_contribution_indices(width, max_h1=max_h1, max_h2=max_h2)

        for key, global_tensor in global_state.items():
            sub_val = sub_state[key]
            indices = indices_map[key]

            expected_shape = tuple(global_tensor[indices].shape)
            if tuple(sub_val.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for key={key!r}, width={width}: "
                    f"expected sub-tensor shape {expected_shape}, got {tuple(sub_val.shape)}. "
                    f"Verify that the client's SuperNet width matches received_width."
                )

            expanded, mask = expand_subtensor_to_global(
                sub_val, indices, tuple(global_tensor.shape)
            )
            weighted_sum[key].add_(expanded, alpha=n)
            count[key].add_(mask, alpha=n)
            client_count[key] += mask  # mask is 0/1, so this counts clients

    _log_aggregation_details(client_count, max_h1, max_h2)

    new_global: dict[str, Tensor] = {}
    for key, global_tensor in global_state.items():
        c = count[key]
        has_contrib = c > 0
        # Avoid division by zero: replace zeros with 1 before dividing
        safe_c = torch.where(has_contrib, c, torch.ones_like(c))
        aggregated = weighted_sum[key] / safe_c
        # Where no client contributed, keep the previous global value
        new_global[key] = torch.where(has_contrib, aggregated, global_tensor.float())

    _logger.debug(
        "aggregate_masked complete | output: %s",
        {k: tuple(v.shape) for k, v in new_global.items()},
    )
    return new_global


def _log_aggregation_details(
    client_count: dict[str, Tensor],
    max_h1: int,
    max_h2: int,
) -> None:
    """Sample 3 fc1.weight row-regions and log their client contributor counts."""
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    if "fc1.weight" not in client_count:
        return

    h1_q = max_h1 // 4   # 64  — weak boundary
    h1_h = max_h1 // 2   # 128 — medium boundary
    fc1_count = client_count["fc1.weight"]

    def _max_in(region: Tensor) -> int:
        return int(region.max().item()) if region.numel() > 0 else 0

    _logger.debug(
        "aggregate_masked | fc1.weight[0:%d,:] : %d contributors | "
        "fc1.weight[%d:%d,:] : %d contributors | "
        "fc1.weight[%d:%d,:] : %d contributors",
        h1_q, _max_in(fc1_count[0:h1_q, :]),
        h1_q, h1_h, _max_in(fc1_count[h1_q:h1_h, :]),
        h1_h, max_h1, _max_in(fc1_count[h1_h:max_h1, :]),
    )
