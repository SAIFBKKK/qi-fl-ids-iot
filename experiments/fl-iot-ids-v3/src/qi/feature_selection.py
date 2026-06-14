from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QGAFeatureSelectionConfig:
    k_features: int = 16
    population_size: int = 24
    generations: int = 20
    mutation_rate: float = 0.08
    crossover_rate: float = 0.8
    redundancy_penalty: float = 0.02
    size_penalty: float = 1.0
    random_seed: int = 42


@dataclass(frozen=True)
class QGAFeatureSelectionResult:
    selected_indices: list[int]
    selected_features: list[str]
    feature_scores: list[float]
    best_fitness: float
    generations: int
    population_size: int
    n_features: int
    k_features: int
    smoke: bool


def ensure_exact_k(mask: np.ndarray, k_features: int, rng: np.random.Generator) -> np.ndarray:
    """Repair a binary solution so it selects exactly k features."""

    repaired = np.asarray(mask, dtype=bool).copy()
    n_features = int(repaired.size)
    k_features = int(k_features)
    if not 1 <= k_features <= n_features:
        raise ValueError(f"k_features must be in [1, {n_features}], got {k_features}.")

    selected = np.flatnonzero(repaired)
    if selected.size > k_features:
        drop = rng.choice(selected, size=selected.size - k_features, replace=False)
        repaired[drop] = False
    elif selected.size < k_features:
        candidates = np.flatnonzero(~repaired)
        add = rng.choice(candidates, size=k_features - selected.size, replace=False)
        repaired[add] = True
    return repaired


def fast_feature_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute lightweight supervised feature scores.

    This is a deterministic filter score used by the QGA fitness. It compares
    between-class variance to within-class variance and does not train a model.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows.")

    classes = np.unique(y)
    global_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1], dtype=np.float64)
    within = np.zeros(X.shape[1], dtype=np.float64)
    for class_id in classes:
        X_c = X[y == class_id]
        if X_c.size == 0:
            continue
        class_mean = X_c.mean(axis=0)
        between += X_c.shape[0] * np.square(class_mean - global_mean)
        within += np.square(X_c - class_mean).sum(axis=0)

    scores = between / np.maximum(within, 1e-12)
    max_score = float(np.max(scores)) if scores.size else 0.0
    if max_score > 0.0:
        scores = scores / max_score
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _redundancy(mask: np.ndarray, X: np.ndarray) -> float:
    selected = np.flatnonzero(mask)
    if selected.size <= 1:
        return 0.0
    X_selected = np.asarray(X[:, selected], dtype=np.float64)
    corr = np.corrcoef(X_selected, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(upper.mean()) if upper.size else 0.0


def fitness(
    mask: np.ndarray,
    *,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    X_val: np.ndarray,
    config: QGAFeatureSelectionConfig,
) -> float:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return -float("inf")

    score = 0.7 * float(train_scores[selected].mean())
    score += 0.3 * float(val_scores[selected].mean())
    score -= float(config.redundancy_penalty) * _redundancy(mask, X_val)
    score -= float(config.size_penalty) * abs(selected.size - config.k_features) / mask.size
    return float(score)


def _initial_population(
    n_features: int,
    config: QGAFeatureSelectionConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    return [
        ensure_exact_k(
            rng.random(n_features) < (config.k_features / n_features),
            config.k_features,
            rng,
        )
        for _ in range(config.population_size)
    ]


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    config: QGAFeatureSelectionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() > config.crossover_rate:
        return parent_a.copy()
    selector = rng.random(parent_a.size) < 0.5
    child = np.where(selector, parent_a, parent_b)
    return ensure_exact_k(child, config.k_features, rng)


def _mutate(
    mask: np.ndarray,
    config: QGAFeatureSelectionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mutated = mask.copy()
    if rng.random() <= config.mutation_rate:
        selected = np.flatnonzero(mutated)
        unselected = np.flatnonzero(~mutated)
        if selected.size and unselected.size:
            mutated[rng.choice(selected)] = False
            mutated[rng.choice(unselected)] = True
    return ensure_exact_k(mutated, config.k_features, rng)


def run_qga_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    *,
    config: QGAFeatureSelectionConfig,
    smoke: bool = False,
) -> QGAFeatureSelectionResult:
    n_features = int(X_train.shape[1])
    if X_val.shape[1] != n_features:
        raise ValueError("Train and validation feature counts differ.")
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length={len(feature_names)} does not match n_features={n_features}."
        )

    effective_config = config
    if smoke:
        effective_config = QGAFeatureSelectionConfig(
            **{
                **asdict(config),
                "population_size": min(config.population_size, 8),
                "generations": min(config.generations, 4),
            }
        )

    rng = np.random.default_rng(effective_config.random_seed)
    train_scores = fast_feature_scores(X_train, y_train)
    val_scores = fast_feature_scores(X_val, y_val)
    population = _initial_population(n_features, effective_config, rng)

    best_mask = population[0]
    best_fitness = -float("inf")
    for _ in range(effective_config.generations):
        scored = [
            (
                fitness(
                    mask,
                    train_scores=train_scores,
                    val_scores=val_scores,
                    X_val=X_val,
                    config=effective_config,
                ),
                mask,
            )
            for mask in population
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_fitness:
            best_fitness = float(scored[0][0])
            best_mask = scored[0][1].copy()

        elite_count = max(2, effective_config.population_size // 4)
        elites = [mask for _, mask in scored[:elite_count]]
        next_population = [mask.copy() for mask in elites]
        while len(next_population) < effective_config.population_size:
            parents = rng.choice(len(elites), size=2, replace=True)
            child = _crossover(elites[int(parents[0])], elites[int(parents[1])], effective_config, rng)
            next_population.append(_mutate(child, effective_config, rng))
        population = next_population

    selected_indices = [int(idx) for idx in np.flatnonzero(best_mask)]
    combined_scores = (0.7 * train_scores) + (0.3 * val_scores)
    return QGAFeatureSelectionResult(
        selected_indices=selected_indices,
        selected_features=[feature_names[idx] for idx in selected_indices],
        feature_scores=[float(combined_scores[idx]) for idx in selected_indices],
        best_fitness=float(best_fitness),
        generations=int(effective_config.generations),
        population_size=int(effective_config.population_size),
        n_features=n_features,
        k_features=int(effective_config.k_features),
        smoke=bool(smoke),
    )


def save_feature_selection_artifacts(
    result: QGAFeatureSelectionResult,
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
        "n_features": result.n_features,
        "k_features": result.k_features,
        "selected_indices": result.selected_indices,
        "selected_features": result.selected_features,
        "feature_scores": result.feature_scores,
        "best_fitness": result.best_fitness,
        "generations": result.generations,
        "population_size": result.population_size,
        "smoke": result.smoke,
        "method": "quantum-inspired genetic algorithm over binary feature masks",
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
    rows = [
        "# QGA Feature Selection Report",
        "",
        f"- Scenario: {payload['scenario']}",
        f"- Input features: {payload['n_features']}",
        f"- Selected features: {payload['k_features']}",
        f"- Best fitness: {payload['best_fitness']:.6f}",
        f"- Smoke mode: {payload['smoke']}",
        "",
        "## Selected Features",
        "",
    ]
    for rank, (idx, name, score) in enumerate(
        zip(
            payload["selected_indices"],
            payload["selected_features"],
            payload["feature_scores"],
        ),
        start=1,
    ):
        rows.append(f"{rank}. `{idx}` - `{name}` - score={float(score):.6f}")

    rows.extend(
        [
            "",
            "This is a quantum-inspired optimisation routine: it uses binary masks,",
            "population evolution, crossover and mutation inspired by QGA-style search.",
            "It does not execute on quantum hardware.",
            "",
        ]
    )
    return "\n".join(rows)
