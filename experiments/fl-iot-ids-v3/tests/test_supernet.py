from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.model.supernet import (
    SuperNet,
    count_parameters,
    extract_submodel_state,
    load_submodel,
)


def _deterministic_full_state() -> dict[str, torch.Tensor]:
    model = SuperNet(width=1.0, dropout=0.0)
    state = model.state_dict()
    return {
        key: torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) / 1000.0
        for key, value in state.items()
    }


def test_full_width_parameter_count():
    assert count_parameters(1.0) == 44_706


def test_medium_width_parameter_count():
    assert count_parameters(0.5) == 14_178


def test_weak_width_parameter_count():
    assert count_parameters(0.25) == 5_058


def test_forward_shape_for_all_widths():
    x = torch.randn(4, 28)

    for width in (0.25, 0.5, 1.0):
        model = SuperNet(width=width)
        y = model(x)

        assert y.shape == (4, 34)


def test_extract_medium_submodel_shapes():
    sub_state = extract_submodel_state(_deterministic_full_state(), 0.5)

    assert sub_state["fc1.weight"].shape == (128, 28)
    assert sub_state["fc1.bias"].shape == (128,)
    assert sub_state["fc2.weight"].shape == (64, 128)
    assert sub_state["fc2.bias"].shape == (64,)
    assert sub_state["fc3.weight"].shape == (34, 64)
    assert sub_state["fc3.bias"].shape == (34,)


def test_extracted_state_loads_into_matching_submodel():
    sub_state = extract_submodel_state(_deterministic_full_state(), 0.5)
    model = SuperNet(width=0.5)

    load_submodel(model, sub_state)

    assert model.fc1.weight.shape == (128, 28)
    assert model.fc2.weight.shape == (64, 128)
    assert model.fc3.weight.shape == (34, 64)


def test_round_trip_forward_matches_manual_extracted_weights():
    sub_state = extract_submodel_state(_deterministic_full_state(), 0.5)
    model = load_submodel(SuperNet(width=0.5, dropout=0.0), sub_state)
    model.eval()
    x = torch.randn(3, 28)

    with torch.no_grad():
        expected = F.linear(x, sub_state["fc1.weight"], sub_state["fc1.bias"])
        expected = F.relu(expected)
        expected = F.linear(expected, sub_state["fc2.weight"], sub_state["fc2.bias"])
        expected = F.relu(expected)
        expected = F.linear(expected, sub_state["fc3.weight"], sub_state["fc3.bias"])
        actual = model(x)

    assert torch.allclose(actual, expected)


def test_invalid_width_raises_assertion_error():
    with pytest.raises(AssertionError):
        SuperNet(width=0.75)
