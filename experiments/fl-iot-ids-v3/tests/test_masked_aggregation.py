from __future__ import annotations

import torch

from src.fl.masked_aggregation import (
    aggregate_masked,
    expand_subtensor_to_global,
)
from src.model.supernet import SuperNet, extract_submodel_state, load_submodel


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _full_state(fill: float) -> dict[str, torch.Tensor]:
    """Full SuperNet(width=1.0) state with every parameter = fill."""
    net = SuperNet(width=1.0, dropout=0.0)
    return {k: torch.full_like(v, fill, dtype=torch.float32) for k, v in net.state_dict().items()}


def _substate(fill: float, width: float) -> dict[str, torch.Tensor]:
    """Sub-state for given width with every parameter = fill."""
    return extract_submodel_state(_full_state(fill), width)


def _fedavg(states: list[dict], n_examples: list[int]) -> dict[str, torch.Tensor]:
    """Reference FedAvg: plain weighted average (all positions always full)."""
    total = float(sum(n_examples))
    result: dict[str, torch.Tensor] = {}
    for key in states[0]:
        result[key] = sum(
            n * s[key].float() for n, s in zip(n_examples, states)
        ) / total
    return result


# ─── INV-1 ────────────────────────────────────────────────────────────────────

def test_inv1_single_powerful_client_is_identity():
    """aggregate_masked with 1 client (width=1.0) returns that client's state."""
    global_state = _full_state(0.5)
    sub = _substate(2.0, 1.0)

    result = aggregate_masked(
        [{"state_dict": sub, "num_examples": 100, "width": 1.0}],
        global_state,
    )

    for key in result:
        assert torch.allclose(result[key], sub[key].float(), atol=1e-6), (
            f"INV-1 failed at {key}"
        )


# ─── INV-2 ────────────────────────────────────────────────────────────────────

def test_inv2_all_powerful_equal_examples_is_arithmetic_mean():
    """3 clients all width=1.0, equal num_examples → plain arithmetic mean."""
    global_state = _full_state(0.0)
    fills = [1.0, 3.0, 5.0]
    states = [_substate(f, 1.0) for f in fills]
    updates = [{"state_dict": s, "num_examples": 100, "width": 1.0} for s in states]

    result = aggregate_masked(updates, global_state)

    for key in result:
        expected = (states[0][key] + states[1][key] + states[2][key]).float() / 3.0
        assert torch.allclose(result[key], expected, atol=1e-5), (
            f"INV-2 failed at {key}: got {result[key].mean():.4f}, expected {expected.mean():.4f}"
        )


# ─── INV-3 ────────────────────────────────────────────────────────────────────

def test_inv3_mixed_widths_weighted_average_per_position():
    """3 clients widths=[0.25, 0.5, 1.0], n=[100, 200, 300] — verify per-region averages."""
    global_state = _full_state(0.0)
    sub_weak     = _substate(1.0, 0.25)   # all 1.0
    sub_medium   = _substate(2.0, 0.5)    # all 2.0
    sub_powerful = _substate(3.0, 1.0)    # all 3.0

    updates = [
        {"state_dict": sub_weak,     "num_examples": 100, "width": 0.25},
        {"state_dict": sub_medium,   "num_examples": 200, "width": 0.5},
        {"state_dict": sub_powerful, "num_examples": 300, "width": 1.0},
    ]

    result = aggregate_masked(updates, global_state)
    fc1 = result["fc1.weight"]

    # fc1[0:64,:]   : 3 contributors → (100×1 + 200×2 + 300×3) / 600 = 7/3
    expected_all = (100 * 1.0 + 200 * 2.0 + 300 * 3.0) / 600.0
    assert torch.allclose(fc1[0:64, :], torch.full((64, 28), expected_all), atol=1e-5), (
        f"INV-3 region [0:64]: expected {expected_all:.4f}, got {fc1[0:64,:].mean():.4f}"
    )

    # fc1[64:128,:] : 2 contributors → (200×2 + 300×3) / 500 = 2.6
    expected_med_pow = (200 * 2.0 + 300 * 3.0) / 500.0
    assert torch.allclose(fc1[64:128, :], torch.full((64, 28), expected_med_pow), atol=1e-5), (
        f"INV-3 region [64:128]: expected {expected_med_pow:.4f}, got {fc1[64:128,:].mean():.4f}"
    )

    # fc1[128:256,:] : 1 contributor → 3.0
    assert torch.allclose(fc1[128:256, :], torch.full((128, 28), 3.0), atol=1e-5), (
        f"INV-3 region [128:256]: expected 3.0, got {fc1[128:256,:].mean():.4f}"
    )


# ─── INV-4 ────────────────────────────────────────────────────────────────────

def test_inv4_no_contributor_retains_global_state():
    """Positions with no contributing client keep their global_state value (no NaN)."""
    sentinel = 5.0
    global_state = _full_state(sentinel)
    sub_weak = _substate(1.0, 0.25)

    updates = [
        {"state_dict": sub_weak, "num_examples": 100, "width": 0.25},
        {"state_dict": sub_weak, "num_examples": 200, "width": 0.25},
    ]

    result = aggregate_masked(updates, global_state)

    # Positions fc1[64:256, :] have no contributor → must retain sentinel
    assert torch.allclose(
        result["fc1.weight"][64:256, :], torch.full((192, 28), sentinel)
    ), "INV-4: uncovered positions were not kept from global_state"

    # Positions fc2[32:128, :] have no contributor
    assert torch.allclose(
        result["fc2.weight"][32:128, :], torch.full((96, 256), sentinel)
    ), "INV-4: fc2 uncovered positions were not kept"

    # No NaN or Inf anywhere
    for key, v in result.items():
        assert torch.isfinite(v).all(), f"INV-4: non-finite values in {key}"


# ─── INV-5 ────────────────────────────────────────────────────────────────────

def test_inv5_expand_subtensor_to_global():
    """expand_subtensor_to_global places sub at correct indices, zeros elsewhere."""
    sub = torch.ones(128, 28) * 7.0
    indices = (slice(0, 128), slice(None))
    global_shape = (256, 28)

    expanded, mask = expand_subtensor_to_global(sub, indices, global_shape)

    assert expanded.shape == global_shape
    assert mask.shape == global_shape
    assert torch.allclose(expanded[0:128, :], torch.full((128, 28), 7.0))
    assert torch.allclose(expanded[128:256, :], torch.zeros(128, 28))
    assert torch.allclose(mask[0:128, :], torch.ones(128, 28))
    assert torch.allclose(mask[128:256, :], torch.zeros(128, 28))


def test_inv5_expand_2d_asymmetric_indices():
    """expand_subtensor_to_global handles asymmetric 2D slices (fc2 case)."""
    sub = torch.ones(32, 64) * 3.0
    indices = (slice(0, 32), slice(0, 64))
    global_shape = (128, 256)

    expanded, mask = expand_subtensor_to_global(sub, indices, global_shape)

    assert expanded.shape == global_shape
    assert torch.allclose(expanded[0:32, 0:64], torch.full((32, 64), 3.0))
    assert torch.allclose(expanded[32:128, :], torch.zeros(96, 256))
    assert torch.allclose(expanded[0:32, 64:256], torch.zeros(32, 192))
    assert torch.allclose(mask[0:32, 0:64], torch.ones(32, 64))
    assert torch.allclose(mask[32:128, :], torch.zeros(96, 256))


# ─── INV-6 ────────────────────────────────────────────────────────────────────

def test_inv6_output_shapes_always_match_global():
    """aggregate_masked always returns full SuperNet shapes regardless of input width."""
    global_state = _full_state(0.0)
    expected_shapes = {k: tuple(v.shape) for k, v in global_state.items()}

    for width in (0.25, 0.5, 1.0):
        sub = _substate(1.0, width)
        updates = [{"state_dict": sub, "num_examples": 100, "width": width}]
        result = aggregate_masked(updates, global_state)

        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(result[key].shape)
            assert actual_shape == expected_shape, (
                f"INV-6 width={width} key={key}: expected {expected_shape}, got {actual_shape}"
            )


# ─── INV-7 ────────────────────────────────────────────────────────────────────

def test_inv7_numerical_stability_extreme_num_examples():
    """No NaN or Inf even with extreme num_examples imbalance (1 vs 1_000_000)."""
    global_state = _full_state(0.0)
    updates = [
        {"state_dict": _substate(0.1, 0.25), "num_examples": 1,          "width": 0.25},
        {"state_dict": _substate(0.1, 1.0),  "num_examples": 1_000_000,  "width": 1.0},
    ]

    result = aggregate_masked(updates, global_state)

    for key, v in result.items():
        assert torch.isfinite(v).all(), f"INV-7: non-finite values in {key}"
        assert not torch.isnan(v).any(), f"INV-7: NaN in {key}"


# ─── INV-8 ────────────────────────────────────────────────────────────────────

def test_inv8_round_trip_extract_from_aggregated_result():
    """extract_submodel_state on aggregate_masked result loads into SuperNet(0.5) cleanly."""
    global_state = _full_state(1.0)
    updates = [
        {"state_dict": _substate(2.0, 0.5),  "num_examples": 100, "width": 0.5},
        {"state_dict": _substate(3.0, 1.0),  "num_examples": 100, "width": 1.0},
    ]

    result = aggregate_masked(updates, global_state)

    # Extract medium sub-state from aggregated result
    extracted = extract_submodel_state(result, 0.5)
    model = SuperNet(width=0.5, dropout=0.0)
    load_submodel(model, extracted)   # must not raise

    # Forward pass must be finite
    x = torch.randn(4, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 34)
    assert torch.isfinite(out).all(), "INV-8: non-finite output after round-trip"


# ─── INV-9 ────────────────────────────────────────────────────────────────────

def test_inv9_round1_all_full_width_equals_fedavg():
    """
    Round-1 simulation: all received_width=1.0 (warm-up full SuperNet).
    aggregate_masked must be mathematically identical to FedAvg.
    tier_width extra key in updates is ignored.
    """
    global_state = _full_state(0.0)
    fills = [1.0, 2.0, 3.0]
    n_examples = [100, 200, 300]
    tier_widths = [0.25, 0.5, 1.0]   # permanent tiers — should NOT affect result
    states = [_substate(f, 1.0) for f in fills]

    updates = [
        {
            "state_dict": s,
            "num_examples": n,
            "width": 1.0,           # received_width = full for all (round 1)
            "tier_width": tw,       # permanent tier — aggregate_masked ignores this
        }
        for s, n, tw in zip(states, n_examples, tier_widths)
    ]

    result = aggregate_masked(updates, global_state)
    expected = _fedavg(states, n_examples)

    for key in result:
        assert torch.allclose(result[key], expected[key].float(), atol=1e-5), (
            f"INV-9 failed at {key}: round-1 result differs from FedAvg"
        )
