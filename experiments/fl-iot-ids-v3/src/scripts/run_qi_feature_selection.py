from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.common.paths import ARTIFACTS_DIR, DATA_DIR, get_processed_path
from src.qi.feature_selection import (
    QGAFeatureSelectionConfig,
    run_qga_feature_selection,
    save_feature_selection_artifacts,
)


DEFAULT_NODE_IDS = ("node1", "node2", "node3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QGA feature selection for v3 scenarios.")
    parser.add_argument("--scenario", required=True, help="Scenario name, e.g. normal_noniid.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--smoke", action="store_true", help="Use a small deterministic run.")
    parser.add_argument(
        "--node-ids",
        nargs="*",
        default=list(DEFAULT_NODE_IDS),
        help="Node IDs to include.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return X, y
    indices = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[indices], y[indices]


def _load_split(
    scenario: str,
    split: str,
    node_ids: list[str],
    max_rows_per_node: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_parts = []
    y_parts = []
    feature_names: list[str] | None = None
    for node_id in node_ids:
        path = get_processed_path(scenario, node_id, split=split)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        X_node, y_node = _sample_rows(data["X"], data["y"], max_rows_per_node, rng)
        X_parts.append(X_node.astype(np.float32, copy=False))
        y_parts.append(y_node.astype(np.int64, copy=False))
        names = [str(name) for name in data["feature_names"].tolist()]
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names differ across nodes; mismatch at {path}")

    if feature_names is None:
        raise ValueError("No feature names loaded.")
    return np.vstack(X_parts), np.concatenate(y_parts), feature_names


def _synthetic_smoke_dataset(config: QGAFeatureSelectionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(config.random_seed)
    n_features = 28
    X_train = rng.normal(size=(96, n_features)).astype(np.float32)
    y_train = rng.integers(0, 4, size=96, dtype=np.int64)
    X_val = rng.normal(size=(48, n_features)).astype(np.float32)
    y_val = rng.integers(0, 4, size=48, dtype=np.int64)
    signal_features = np.arange(min(config.k_features, n_features))
    X_train[:, signal_features] += y_train[:, None] * 0.25
    X_val[:, signal_features] += y_val[:, None] * 0.25
    return X_train, y_train, X_val, y_val, [f"feature_{idx}" for idx in range(n_features)]


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    qga_cfg = dict(raw_cfg.get("qga_feature_selection", raw_cfg))
    config = QGAFeatureSelectionConfig(
        k_features=int(qga_cfg.get("k_features", 16)),
        population_size=int(qga_cfg.get("population_size", 24)),
        generations=int(qga_cfg.get("generations", 20)),
        mutation_rate=float(qga_cfg.get("mutation_rate", 0.08)),
        crossover_rate=float(qga_cfg.get("crossover_rate", 0.8)),
        redundancy_penalty=float(qga_cfg.get("redundancy_penalty", 0.02)),
        size_penalty=float(qga_cfg.get("size_penalty", 1.0)),
        random_seed=int(qga_cfg.get("random_seed", 42)),
    )
    smoke_cfg = dict(qga_cfg.get("smoke", {}))
    max_rows_per_node = int(
        smoke_cfg.get("max_rows_per_node", 512)
        if args.smoke
        else qga_cfg.get("max_rows_per_node", 5000)
    )
    allow_synthetic_smoke = bool(smoke_cfg.get("allow_synthetic_if_missing", True))

    rng = np.random.default_rng(config.random_seed)
    try:
        X_train, y_train, feature_names = _load_split(
            args.scenario,
            "train",
            args.node_ids,
            max_rows_per_node,
            rng,
        )
        X_val, y_val, val_feature_names = _load_split(
            args.scenario,
            "val",
            args.node_ids,
            max_rows_per_node,
            rng,
        )
        if feature_names != val_feature_names:
            raise ValueError("Train and validation feature names differ.")
    except FileNotFoundError:
        if not args.smoke or not allow_synthetic_smoke:
            raise
        X_train, y_train, X_val, y_val, feature_names = _synthetic_smoke_dataset(config)

    result = run_qga_feature_selection(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        config=config,
        smoke=bool(args.smoke),
    )
    output_dir = ARTIFACTS_DIR / "qi_feature_selection" / args.scenario
    paths = save_feature_selection_artifacts(
        result,
        output_dir=output_dir,
        scenario=args.scenario,
    )
    print(f"Selected {len(result.selected_indices)} / {result.n_features} features")
    print(f"selected_features={paths['selected_features']}")
    print(f"feature_mask={paths['feature_mask']}")
    print(f"selection_report={paths['selection_report']}")


if __name__ == "__main__":
    main()
