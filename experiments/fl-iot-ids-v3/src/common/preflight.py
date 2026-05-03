from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common.paths import ARTIFACTS_DIR, DATA_DIR


def validate_required_artifacts(
    config: Mapping[str, Any],
    node_ids: list[str],
    *,
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    scenario = str(config.get("scenario", {}).get("name", "normal_noniid"))
    manifest_path = data_dir / "splits" / f"{scenario}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing scenario manifest: {manifest_path}. "
            f"Run: python -m src.scripts.generate_scenarios --scenario {scenario}"
        )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" not in manifest:
        raise ValueError(
            f"Legacy manifest detected for scenario={scenario!r}: {manifest_path}. "
            "Regenerate the scenario with the split-aware pipeline: "
            f"python -m src.scripts.generate_scenarios --scenario {scenario}"
        )

    missing_paths: list[str] = []
    for node_id in node_ids:
        for split in ("train", "val"):
            split_path = data_dir / "processed" / scenario / node_id / f"{split}_preprocessed.npz"
            if not split_path.exists():
                missing_paths.append(str(split_path))

    imbalance_name = str(
        config.get("imbalance", {}).get(
            "name",
            config.get("imbalance_strategy", ""),
        )
    ).lower()
    if imbalance_name in {"class_weights", "focal_loss_weighted"}:
        class_weights_path = artifacts_dir / f"class_weights_{scenario}.pkl"
        if not class_weights_path.exists():
            raise FileNotFoundError(
                f"Missing scenario-specific class weights: {class_weights_path}. "
                f"Run: python -m src.scripts.generate_weights --scenario {scenario}"
            )

    model_cfg = dict(config.get("model", {}))
    feature_cfg = dict(config.get("feature_selection", {}))
    feature_cfg.update(dict(model_cfg.get("feature_selection", {})))
    if bool(feature_cfg.get("enabled", False)):
        raw_path = str(
            feature_cfg.get(
                "artifact_path",
                "artifacts/qi_feature_selection/{scenario}/selected_features.json",
            )
        ).format(scenario=scenario)
        feature_path = Path(raw_path)
        if not feature_path.is_absolute():
            feature_path = data_dir.parent / feature_path
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing QGA selected-features artifact: {feature_path}. "
                "Run: python -m src.scripts.run_qi_feature_selection "
                f"--scenario {scenario} --config configs/qi/qga_feature_selection.yaml"
            )

    if missing_paths:
        joined = ", ".join(missing_paths)
        raise FileNotFoundError(
            "Explicit train/val NPZ splits are required before FL simulation. "
            f"Missing artifacts: {joined}. "
            f"Run: python -m src.scripts.generate_scenarios --scenario {scenario}"
        )
