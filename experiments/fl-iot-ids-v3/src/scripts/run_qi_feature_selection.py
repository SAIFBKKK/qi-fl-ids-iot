from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from src.common.paths import ARTIFACTS_DIR, get_processed_path
from src.qi.qi_feature_selector import (
    QIFeatureSelectorConfig,
    run_qi_feature_selection,
    save_qi_feature_selection_artifacts,
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


def _load_split(
    scenario: str,
    split: str,
    node_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_parts = []
    y_parts = []
    feature_names: list[str] | None = None
    for node_id in node_ids:
        path = get_processed_path(scenario, node_id, split=split)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        X_parts.append(data["X"].astype(np.float32, copy=False))
        y_parts.append(data["y"].astype(np.int64, copy=False))
        names = [str(name) for name in data["feature_names"].tolist()]
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names differ across nodes; mismatch at {path}")

    if feature_names is None:
        raise ValueError("No feature names loaded.")
    return np.vstack(X_parts), np.concatenate(y_parts), feature_names


def _synthetic_smoke_dataset(
    config: QIFeatureSelectorConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(config.seed)
    n_features = int(config.n_features)
    rows = max(96, int(config.max_samples_per_class) * 4)
    X = rng.normal(size=(rows, n_features)).astype(np.float32)
    y = rng.integers(0, 4, size=rows, dtype=np.int64)
    signal_features = np.arange(min(config.k_features, n_features))
    X[:, signal_features] += y[:, None] * 0.35
    return X, y, [f"feature_{idx}" for idx in range(n_features)]


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    qga_cfg = dict(raw_cfg.get("qga_feature_selection", raw_cfg))
    mode = "smoke" if args.smoke else str(qga_cfg.get("mode", "full"))
    smoke_cfg = dict(qga_cfg.get("smoke", {}))
    active_cfg = {**qga_cfg, **(smoke_cfg if args.smoke else {})}
    config = QIFeatureSelectorConfig(
        n_features=int(active_cfg.get("n_features", 28)),
        k_features=int(active_cfg.get("k_features", 15)),
        n_generations=int(active_cfg.get("n_generations", active_cfg.get("generations", 12))),
        pop_size=int(active_cfg.get("pop_size", active_cfg.get("population_size", 12))),
        epochs=int(active_cfg.get("epochs", 2)),
        max_samples_per_class=int(active_cfg.get("max_samples_per_class", 40)),
        seed=int(active_cfg.get("seed", active_cfg.get("random_seed", 42))),
        mode=mode,
        theta_update_rate=float(active_cfg.get("theta_update_rate", 0.12)),
        size_penalty=float(active_cfg.get("size_penalty", 0.0)),
    )
    allow_synthetic_smoke = bool(smoke_cfg.get("allow_synthetic_if_missing", True))

    try:
        X, y, feature_names = _load_split(
            args.scenario,
            "train",
            args.node_ids,
        )
    except FileNotFoundError:
        if not args.smoke or not allow_synthetic_smoke:
            raise
        X, y, feature_names = _synthetic_smoke_dataset(config)

    result = run_qi_feature_selection(
        X,
        y,
        feature_names,
        config=config,
        num_classes=int(qga_cfg.get("num_classes", 34)),
    )
    output_dir = ARTIFACTS_DIR / "qi_feature_selection" / args.scenario
    paths = save_qi_feature_selection_artifacts(
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
