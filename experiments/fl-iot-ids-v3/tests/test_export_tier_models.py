from __future__ import annotations

import json

import pytest
import torch

from src.model.network import MLPClassifier
from src.model.supernet import SuperNet
from src.scripts.export_tier_models import (
    count_state_dict_parameters,
    export_tier_models,
    load_checkpoint_state_dict,
    tensor_shapes,
    verify_exported_model,
)


EXPECTED_KEYS = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
)

EXPECTED_SHAPES = {
    "weak": {
        "fc1.weight": [64, 28],
        "fc1.bias": [64],
        "fc2.weight": [32, 64],
        "fc2.bias": [32],
        "fc3.weight": [34, 32],
        "fc3.bias": [34],
    },
    "medium": {
        "fc1.weight": [128, 28],
        "fc1.bias": [128],
        "fc2.weight": [64, 128],
        "fc2.bias": [64],
        "fc3.weight": [34, 64],
        "fc3.bias": [34],
    },
    "powerful": {
        "fc1.weight": [256, 28],
        "fc1.bias": [256],
        "fc2.weight": [128, 256],
        "fc2.bias": [128],
        "fc3.weight": [34, 128],
        "fc3.bias": [34],
    },
}


def _full_state() -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in SuperNet(width=1.0, dropout=0.0).state_dict().items()
    }


def _checkpoint_path(tmp_path, payload) -> object:
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    return path


def test_load_checkpoint_state_dict_accepts_direct_state_dict(tmp_path):
    state = _full_state()
    path = _checkpoint_path(tmp_path, state)

    loaded = load_checkpoint_state_dict(path)

    assert tuple(loaded.keys()) == EXPECTED_KEYS
    assert tensor_shapes(loaded)["fc1.weight"] == [256, 28]


def test_load_checkpoint_state_dict_accepts_model_state_dict_wrapper(tmp_path):
    state = _full_state()
    path = _checkpoint_path(tmp_path, {"model_state_dict": state})

    loaded = load_checkpoint_state_dict(path)

    assert tuple(loaded.keys()) == EXPECTED_KEYS
    assert torch.equal(loaded["fc3.bias"], state["fc3.bias"])


def test_refuses_mlpclassifier_checkpoint(tmp_path):
    state = MLPClassifier(input_dim=28, num_classes=34).state_dict()
    path = _checkpoint_path(tmp_path, state)

    with pytest.raises(ValueError, match="MLPClassifier|SuperNet"):
        load_checkpoint_state_dict(path)


def test_exported_weak_shapes_are_correct(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    summary = export_tier_models(path, output_dir)

    assert summary["tiers"]["weak"]["shapes"] == EXPECTED_SHAPES["weak"]


def test_exported_medium_shapes_are_correct(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    summary = export_tier_models(path, output_dir)

    assert summary["tiers"]["medium"]["shapes"] == EXPECTED_SHAPES["medium"]


def test_exported_powerful_shapes_are_correct(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    summary = export_tier_models(path, output_dir)

    assert summary["tiers"]["powerful"]["shapes"] == EXPECTED_SHAPES["powerful"]


def test_metadata_contains_required_fields(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    export_tier_models(path, output_dir)
    metadata = json.loads((output_dir / "weak" / "metadata.json").read_text())

    assert metadata["tier"] == "weak"
    assert metadata["width"] == 0.25
    assert metadata["architecture"] == "28 -> 64 -> 32 -> 34"
    assert metadata["num_parameters"] == 5_058
    assert metadata["shapes"] == EXPECTED_SHAPES["weak"]
    assert metadata["state_dict_keys"] == list(EXPECTED_KEYS)


def test_verify_reload_forward_pass(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    export_tier_models(path, output_dir, verify=True)
    metadata = json.loads((output_dir / "medium" / "metadata.json").read_text())

    verify_exported_model(output_dir / "medium" / "model.pth", metadata)


def test_overwrite_false_protects_existing_output_dir(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_tier_models(path, output_dir, overwrite=False)


def test_count_state_dict_parameters_matches_expected_tier_counts(tmp_path):
    path = _checkpoint_path(tmp_path, {"state_dict": _full_state()})
    output_dir = tmp_path / "exports"

    export_tier_models(path, output_dir)
    for tier, expected in {
        "weak": 5_058,
        "medium": 14_178,
        "powerful": 44_706,
    }.items():
        payload = torch.load(output_dir / tier / "model.pth", map_location="cpu")
        assert count_state_dict_parameters(payload["state_dict"]) == expected
