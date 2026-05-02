from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.network import MLPClassifier


EXPECTED = {
    "weak": [64],
    "medium": [128, 64],
    "powerful": [256, 128],
}
NODE_IDS = ("node1", "node2", "node3")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_npz_sample(scenario: str, split: str) -> tuple[np.ndarray, int]:
    path = ROOT / "data" / "processed" / scenario / "node1" / f"{split}_preprocessed.npz"
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["X"][0:1], dtype=np.float32), int(data["y"][0])


def predict(bundle_dir: Path, sample: np.ndarray) -> int:
    cfg = load_json(bundle_dir / "model_config.json")
    model = MLPClassifier(
        input_dim=int(cfg["input_dim"]),
        num_classes=int(cfg["num_classes"]),
        hidden_dims=[int(v) for v in cfg["hidden_layers"]],
        dropout=0.0,
    )
    state_dict = torch.load(bundle_dir / "global_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(sample, dtype=torch.float32))
    pred = int(torch.argmax(logits, dim=1).item())
    if not 0 <= pred < int(cfg["num_classes"]):
        raise AssertionError(f"prediction out of range: {pred}")
    return pred


def validate_label_mapping(path: Path) -> None:
    payload = load_json(path)
    if len(payload.get("label_to_id", {})) != 34:
        raise AssertionError("label_to_id must contain 34 classes")
    if len(payload.get("id_to_label", {})) != 34:
        raise AssertionError("id_to_label must contain 34 classes")


def validate_disjoint_rows(scenario: str) -> None:
    row_sets: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        values: set[int] = set()
        for node_id in NODE_IDS:
            path = ROOT / "data" / "raw" / scenario / node_id / f"{split}.csv"
            chunk = pd.read_csv(path, usecols=["__row_id"])
            values.update(int(v) for v in chunk["__row_id"].to_numpy())
        row_sets[split] = values

    overlaps = {
        "train_val": row_sets["train"] & row_sets["val"],
        "train_test": row_sets["train"] & row_sets["test"],
        "val_test": row_sets["val"] & row_sets["test"],
    }
    bad = {name: len(values) for name, values in overlaps.items() if values}
    if bad:
        raise AssertionError(f"row_id leakage detected: {bad}")


def validate_bundle(output_root: Path, model_name: str) -> dict:
    bundle_dir = output_root / model_name
    for filename in (
        "global_model.pth",
        "scaler.pkl",
        "feature_names.pkl",
        "label_mapping.json",
        "model_config.json",
        "run_summary.json",
        "metrics.json",
    ):
        assert_file(bundle_dir / filename)

    cfg = load_json(bundle_dir / "model_config.json")
    expected_layers = EXPECTED[model_name]
    if [int(v) for v in cfg["hidden_layers"]] != expected_layers:
        raise AssertionError(f"{model_name} hidden_layers mismatch: {cfg['hidden_layers']}")
    if int(cfg["input_dim"]) != 28 or int(cfg["num_classes"]) != 34:
        raise AssertionError(f"{model_name} model dimensions mismatch")

    with (bundle_dir / "feature_names.pkl").open("rb") as handle:
        feature_names = pickle.load(handle)
    if len(feature_names) != 28:
        raise AssertionError(f"{model_name} feature_names length is {len(feature_names)}")

    with (bundle_dir / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    transformed = scaler.transform(np.zeros((1, 28), dtype=np.float32))
    if transformed.shape != (1, 28):
        raise AssertionError(f"{model_name} scaler returned {transformed.shape}")

    validate_label_mapping(bundle_dir / "label_mapping.json")

    val_sample, _ = load_npz_sample(str(cfg.get("scenario", "normal_noniid")), "val")
    deploy_sample, _ = load_npz_sample(str(cfg.get("scenario", "normal_noniid")), "test")
    val_pred = predict(bundle_dir, val_sample)
    deployment_pred = predict(bundle_dir, deploy_sample)
    return {
        "bundle": str(bundle_dir),
        "validation_prediction": val_pred,
        "deployment_prediction": deployment_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model factory bundles.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output if args.output.is_absolute() else (Path.cwd() / args.output)

    deployment_dir = output_root / "deployment_data"
    assert_file(deployment_dir / "deployment_15.parquet")
    assert_file(deployment_dir / "feature_names.pkl")
    assert_file(deployment_dir / "label_mapping.json")
    split_summary = load_json(deployment_dir / "split_summary.json")
    if split_summary["anti_leakage"]["deployment_used_for_training"]:
        raise AssertionError("deployment split marked as used for training")
    if split_summary["anti_leakage"]["deployment_used_for_validation"]:
        raise AssertionError("deployment split marked as used for validation")

    scenario = str(split_summary.get("scenario", "normal_noniid"))
    validate_disjoint_rows(scenario)

    present_models = [
        name for name in EXPECTED
        if (output_root / name).is_dir()
    ]
    if not present_models:
        raise FileNotFoundError(
            f"No model bundle directory found in {output_root}. "
            f"Expected any of: {', '.join(EXPECTED)}"
        )

    results = {name: validate_bundle(output_root, name) for name in present_models}
    skipped = [name for name in EXPECTED if name not in present_models]
    summary = {
        "status": "ok",
        "output": str(output_root),
        "scenario": scenario,
        "validated_models": present_models,
        "skipped_absent_models": skipped,
        "results": results,
    }
    print("\nMODEL FACTORY BUNDLE VALIDATION")
    print(f"Output: {output_root}")
    print(f"Scenario: {scenario}")
    print(f"Validated: {', '.join(present_models)}")
    if skipped:
        print(f"Skipped absent models: {', '.join(skipped)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
