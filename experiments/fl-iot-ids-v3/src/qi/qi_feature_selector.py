from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.model.network import MLPClassifier


@dataclass(frozen=True)
class QIFeatureSelectorConfig:
    n_features: int = 28
    k_features: int = 15
    n_generations: int = 12
    pop_size: int = 12
    epochs: int = 2
    max_samples_per_class: int = 40
    seed: int = 42
    mode: str = "smoke"
    theta_update_rate: float = 0.12
    size_penalty: float = 0.0
    hidden_dims: tuple[int, int] = (32, 16)
    learning_rate: float = 0.001


@dataclass(frozen=True)
class QIFeatureSelectionResult:
    selected_indices: list[int]
    selected_features: list[str]
    best_fitness: float
    best_macro_f1: float
    n_features: int
    k_features: int
    n_generations: int
    pop_size: int
    epochs: int
    mode: str
    seed: int


def repair_exact_k(
    mask: np.ndarray,
    k_features: int,
    rng: np.random.Generator,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    repaired = np.asarray(mask, dtype=bool).copy()
    n_features = int(repaired.size)
    k_features = int(k_features)
    if not 1 <= k_features <= n_features:
        raise ValueError(f"k_features must be in [1, {n_features}], got {k_features}.")

    selected = np.flatnonzero(repaired)
    if selected.size > k_features:
        if probabilities is None:
            drop = rng.choice(selected, size=selected.size - k_features, replace=False)
        else:
            order = selected[np.argsort(probabilities[selected])]
            drop = order[: selected.size - k_features]
        repaired[drop] = False
    elif selected.size < k_features:
        candidates = np.flatnonzero(~repaired)
        if probabilities is None:
            add = rng.choice(candidates, size=k_features - selected.size, replace=False)
        else:
            order = candidates[np.argsort(probabilities[candidates])[::-1]]
            add = order[: k_features - selected.size]
        repaired[add] = True
    return repaired


def theta_to_probabilities(theta: np.ndarray) -> np.ndarray:
    return np.square(np.sin(theta))


def sample_mask(
    theta: np.ndarray,
    k_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = theta_to_probabilities(theta)
    raw_mask = rng.random(theta.size) < probabilities
    return repair_exact_k(raw_mask, k_features, rng, probabilities)


def update_theta_towards_best(
    theta: np.ndarray,
    best_mask: np.ndarray,
    *,
    learning_rate: float,
) -> np.ndarray:
    target = np.where(best_mask, np.pi / 2.0, 0.0)
    updated = theta + float(learning_rate) * (target - theta)
    return np.clip(updated, 0.01, (np.pi / 2.0) - 0.01)


def balanced_class_sample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples_per_class <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        take = min(int(max_samples_per_class), class_indices.size)
        if take > 0:
            selected_indices.append(rng.choice(class_indices, size=take, replace=False))
    if not selected_indices:
        raise ValueError("Cannot sample from an empty dataset.")

    indices = np.concatenate(selected_indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def _split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    test_size = 0.25
    n_test = int(np.ceil(X.shape[0] * test_size))
    n_train = int(X.shape[0] - n_test)
    stratify = (
        y
        if classes.size > 1
        and int(counts.min()) >= 2
        and n_test >= classes.size
        and n_train >= classes.size
        else None
    )
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def _train_eval_mini_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    config: QIFeatureSelectorConfig,
) -> float:
    torch.manual_seed(int(config.seed))
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    criterion = torch.nn.CrossEntropyLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    model.train()
    for _ in range(max(1, int(config.epochs))):
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_val_tensor), dim=1).cpu().numpy()
    return float(f1_score(y_val, preds, average="macro", zero_division=0))


def evaluate_mask_fitness(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    config: QIFeatureSelectorConfig,
) -> tuple[float, float]:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return -float("inf"), 0.0

    X_selected = X[:, selected]
    X_train, X_val, y_train, y_val = _split_train_validation(
        X_selected,
        y,
        seed=config.seed,
    )
    macro_f1 = _train_eval_mini_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=num_classes,
        config=config,
    )
    penalty = float(config.size_penalty) * abs(selected.size - config.k_features) / mask.size
    return float(macro_f1 - penalty), float(macro_f1)


def run_qi_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    config: QIFeatureSelectorConfig,
    num_classes: int = 34,
) -> QIFeatureSelectionResult:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    if X.shape[1] != int(config.n_features):
        raise ValueError(f"Expected {config.n_features} features, got {X.shape[1]}.")
    if len(feature_names) != int(config.n_features):
        raise ValueError("feature_names length must match config.n_features.")

    X_sampled, y_sampled = balanced_class_sample(
        X,
        y,
        max_samples_per_class=int(config.max_samples_per_class),
        seed=int(config.seed),
    )

    rng = np.random.default_rng(int(config.seed))
    theta = np.full(int(config.n_features), np.pi / 4.0, dtype=np.float64)
    best_mask = repair_exact_k(
        np.zeros(int(config.n_features), dtype=bool),
        int(config.k_features),
        rng,
        theta_to_probabilities(theta),
    )
    best_fitness = -float("inf")
    best_macro_f1 = 0.0
    cache: dict[tuple[int, ...], tuple[float, float]] = {}

    for _ in range(max(1, int(config.n_generations))):
        generation: list[tuple[float, float, np.ndarray]] = []
        for _ in range(max(1, int(config.pop_size))):
            mask = sample_mask(theta, int(config.k_features), rng)
            key = tuple(int(v) for v in mask.astype(np.int8))
            if key not in cache:
                cache[key] = evaluate_mask_fitness(
                    mask,
                    X_sampled,
                    y_sampled,
                    num_classes=num_classes,
                    config=config,
                )
            fitness_value, macro_f1 = cache[key]
            generation.append((fitness_value, macro_f1, mask))

        fitness_value, macro_f1, generation_best = max(
            generation,
            key=lambda item: item[0],
        )
        if fitness_value > best_fitness:
            best_fitness = float(fitness_value)
            best_macro_f1 = float(macro_f1)
            best_mask = generation_best.copy()
        theta = update_theta_towards_best(
            theta,
            best_mask,
            learning_rate=float(config.theta_update_rate),
        )

    selected_indices = [int(idx) for idx in np.flatnonzero(best_mask)]
    return QIFeatureSelectionResult(
        selected_indices=selected_indices,
        selected_features=[feature_names[idx] for idx in selected_indices],
        best_fitness=float(best_fitness),
        best_macro_f1=float(best_macro_f1),
        n_features=int(config.n_features),
        k_features=int(config.k_features),
        n_generations=int(config.n_generations),
        pop_size=int(config.pop_size),
        epochs=int(config.epochs),
        mode=str(config.mode),
        seed=int(config.seed),
    )


def save_qi_feature_selection_artifacts(
    result: QIFeatureSelectionResult,
    *,
    output_dir: Path,
    scenario: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(result.n_features, dtype=bool)
    mask[np.asarray(result.selected_indices, dtype=int)] = True

    selected_json = output_dir / "selected_features.json"
    mask_path = output_dir / "feature_mask.npy"
    report_path = output_dir / "selection_report.md"
    payload: dict[str, Any] = {
        "scenario": scenario,
        "method": "quantum-inspired theta-vector feature selection",
        **asdict(result),
    }

    selected_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(mask_path, mask)
    report_path.write_text(build_selection_report(payload), encoding="utf-8")
    return {
        "selected_features": selected_json,
        "feature_mask": mask_path,
        "selection_report": report_path,
    }


def build_selection_report(payload: dict[str, Any]) -> str:
    lines = [
        "# QGA Feature Selection Report",
        "",
        f"- Scenario: {payload['scenario']}",
        f"- Input features: {payload['n_features']}",
        f"- Selected features: {payload['k_features']}",
        f"- Mode: {payload['mode']}",
        f"- Generations: {payload['n_generations']}",
        f"- Population size: {payload['pop_size']}",
        f"- Mini-MLP epochs: {payload['epochs']}",
        f"- Best validation Macro-F1: {payload['best_macro_f1']:.6f}",
        f"- Best fitness: {payload['best_fitness']:.6f}",
        "",
        "## Selected Features",
        "",
    ]
    for rank, (idx, name) in enumerate(
        zip(payload["selected_indices"], payload["selected_features"]),
        start=1,
    ):
        lines.append(f"{rank}. `{idx}` - `{name}`")
    lines.extend(
        [
            "",
            "This selector is quantum-inspired: it maintains a theta vector of",
            "length 28, samples masks with p_i = sin(theta_i)^2, repairs each mask",
            "to exactly K selected features, and updates theta toward the best mask.",
            "It does not use quantum hardware.",
            "",
        ]
    )
    return "\n".join(lines)
