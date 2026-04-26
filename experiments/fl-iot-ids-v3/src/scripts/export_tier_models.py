from __future__ import annotations

import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.model.supernet import SuperNet, extract_submodel_state, load_submodel


SUPERNET_KEYS = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
)

TIERS: dict[str, dict[str, Any]] = {
    "weak": {"width": 0.25, "hidden_dims": [64, 32]},
    "medium": {"width": 0.5, "hidden_dims": [128, 64]},
    "powerful": {"width": 1.0, "hidden_dims": [256, 128]},
}


def _load_torch_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _coerce_state_dict(candidate: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(candidate, Mapping):
        if all(key in candidate for key in SUPERNET_KEYS):
            return {
                key: value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else torch.tensor(value)
                for key, value in candidate.items()
                if key in SUPERNET_KEYS
            }
        return None

    if hasattr(candidate, "state_dict"):
        state = candidate.state_dict()
        if isinstance(state, Mapping):
            return _coerce_state_dict(state)
    return None


def validate_supernet_state_dict(state_dict: Mapping[str, Any]) -> None:
    keys = set(state_dict.keys())
    if any(key.startswith("net.") for key in keys):
        raise ValueError(
            "Refusing to export an MLPClassifier checkpoint "
            "(found net.* keys). Expected SuperNet fc1/fc2/fc3 keys."
        )

    missing = [key for key in SUPERNET_KEYS if key not in state_dict]
    if missing:
        raise ValueError(
            f"Checkpoint is not a valid SuperNet state_dict. Missing keys: {missing}"
        )

    expected_shapes = {
        "fc1.weight": (256, 28),
        "fc1.bias": (256,),
        "fc2.weight": (128, 256),
        "fc2.bias": (128,),
        "fc3.weight": (34, 128),
        "fc3.bias": (34,),
    }
    actual_shapes = {
        key: tuple(value.shape)
        for key, value in state_dict.items()
        if key in expected_shapes and hasattr(value, "shape")
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Unexpected SuperNet full-state shapes. "
            f"Expected {expected_shapes}, got {actual_shapes}."
        )


def load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return a validated full-width SuperNet state_dict."""
    checkpoint = _load_torch_checkpoint(checkpoint_path)

    candidates: list[Any] = [checkpoint]
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                candidates.append(checkpoint[key])

    for candidate in candidates:
        state_dict = _coerce_state_dict(candidate)
        if state_dict is not None:
            validate_supernet_state_dict(state_dict)
            return {key: state_dict[key] for key in SUPERNET_KEYS}

    if isinstance(checkpoint, Mapping) and any(
        str(key).startswith("net.") for key in checkpoint
    ):
        raise ValueError(
            "Refusing to export an MLPClassifier state_dict. Expected SuperNet."
        )

    raise ValueError(
        "Could not find a valid SuperNet state_dict in checkpoint. "
        "Supported formats: direct state_dict, or wrapper keys "
        "'state_dict', 'model_state_dict', 'model'."
    )


def load_model_config(model_config_path: Path | None) -> dict[str, Any]:
    defaults = {
        "input_dim": SuperNet.INPUT_DIM,
        "output_dim": SuperNet.OUTPUT_DIM,
        "max_hidden_1": SuperNet.MAX_HIDDEN_1,
        "max_hidden_2": SuperNet.MAX_HIDDEN_2,
        "dropout": 0.2,
    }
    if model_config_path is None:
        return defaults

    with model_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    model_cfg = payload.get("model", payload)
    if not isinstance(model_cfg, Mapping):
        raise ValueError(f"Invalid model config: {model_config_path}")

    defaults.update(
        {
            "input_dim": int(model_cfg.get("input_dim", defaults["input_dim"])),
            "output_dim": int(model_cfg.get("output_dim", defaults["output_dim"])),
            "max_hidden_1": int(
                model_cfg.get("max_hidden_1", defaults["max_hidden_1"])
            ),
            "max_hidden_2": int(
                model_cfg.get("max_hidden_2", defaults["max_hidden_2"])
            ),
            "dropout": float(model_cfg.get("dropout", defaults["dropout"])),
        }
    )
    return defaults


def tensor_shapes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in state_dict.items()}


def count_state_dict_parameters(state_dict: Mapping[str, torch.Tensor]) -> int:
    return int(sum(value.numel() for value in state_dict.values()))


def architecture_string(input_dim: int, hidden_dims: list[int], output_dim: int) -> str:
    dims = [input_dim, *hidden_dims, output_dim]
    return " -> ".join(str(dim) for dim in dims)


def build_metadata(
    *,
    tier: str,
    width: float,
    state_dict: Mapping[str, torch.Tensor],
    source_checkpoint: Path,
    model_config_path: Path | None,
    model_config: Mapping[str, Any],
    exported_at: str,
) -> dict[str, Any]:
    hidden_dims = [
        int(int(model_config["max_hidden_1"]) * width),
        int(int(model_config["max_hidden_2"]) * width),
    ]
    input_dim = int(model_config["input_dim"])
    output_dim = int(model_config["output_dim"])
    return {
        "tier": tier,
        "width": float(width),
        "architecture": architecture_string(input_dim, hidden_dims, output_dim),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "num_parameters": count_state_dict_parameters(state_dict),
        "source_checkpoint": str(source_checkpoint),
        "model_config": str(model_config_path) if model_config_path else None,
        "exported_at": exported_at,
        "state_dict_keys": list(state_dict.keys()),
        "shapes": tensor_shapes(state_dict),
    }


def verify_exported_model(model_path: Path, metadata: Mapping[str, Any]) -> None:
    payload = _load_torch_checkpoint(model_path)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError(f"Invalid exported model payload: {model_path}")

    model = SuperNet(
        width=float(metadata["width"]),
        dropout=0.0,
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
    )
    load_submodel(model, payload["state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, int(metadata["input_dim"])))
    expected_shape = (2, int(metadata["output_dim"]))
    if tuple(output.shape) != expected_shape:
        raise ValueError(
            f"Forward verification failed for {model_path}: "
            f"expected {expected_shape}, got {tuple(output.shape)}"
        )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def export_tier_models(
    checkpoint_path: Path,
    output_dir: Path,
    model_config_path: Path | None = None,
    overwrite: bool = False,
    verify: bool = False,
) -> dict[str, Any]:
    """Export weak, medium and powerful SuperNet sub-model bundles."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    model_config_path = Path(model_config_path) if model_config_path else None

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    model_config = load_model_config(model_config_path)
    _prepare_output_dir(output_dir, overwrite=overwrite)

    exported_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "source_checkpoint": str(checkpoint_path),
        "model_config": str(model_config_path) if model_config_path else None,
        "export_dir": str(output_dir),
        "exported_at": exported_at,
        "tiers": {},
    }

    for tier, spec in TIERS.items():
        width = float(spec["width"])
        tier_dir = output_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)

        sub_state = extract_submodel_state(state_dict, width)
        metadata = build_metadata(
            tier=tier,
            width=width,
            state_dict=sub_state,
            source_checkpoint=checkpoint_path,
            model_config_path=model_config_path,
            model_config=model_config,
            exported_at=exported_at,
        )

        model_path = tier_dir / "model.pth"
        metadata_path = tier_dir / "metadata.json"
        torch.save(
            {
                "tier": tier,
                "width": width,
                "architecture": "SuperNet",
                "input_dim": metadata["input_dim"],
                "hidden_dims": metadata["hidden_dims"],
                "output_dim": metadata["output_dim"],
                "state_dict": sub_state,
                "source_checkpoint": str(checkpoint_path),
                "exported_at": exported_at,
            },
            model_path,
        )
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if verify:
            verify_exported_model(model_path, metadata)

        summary["tiers"][tier] = {
            "width": width,
            "architecture": metadata["architecture"],
            "num_parameters": metadata["num_parameters"],
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "shapes": metadata["shapes"],
        }

    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tier-specific SuperNet models from a full checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_tier_models(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_config_path=args.model_config,
        overwrite=bool(args.overwrite),
        verify=bool(args.verify),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
