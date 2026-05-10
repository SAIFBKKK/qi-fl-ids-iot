"""Quantum-inspired genetic feature selector for P8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from qga.chromosome import (
    mutate_mask,
    sample_quantum_mask,
    update_theta_towards_best,
)
from qga.fitness import compute_qga_fitness


@dataclass
class QGAResult:
    best_mask: np.ndarray
    best_metrics: dict[str, Any]
    best_fitness: float
    history: list[dict[str, Any]]
    feature_ranking: list[dict[str, Any]]
    theta_probabilities: list[float]


def run_qga(
    *,
    n_features: int,
    params: dict[str, Any],
    evaluate_mask: Callable[[np.ndarray, int], dict[str, Any]],
) -> QGAResult:
    seed = int(params["seed"])
    population_size = int(params["population_size"])
    generations = int(params["generations"])
    mutation_rate = float(params["mutation_rate"])
    min_features = int(params["min_features"])
    max_features = int(params["max_features"])
    weights = dict(params["weights"])
    rng = np.random.default_rng(seed)
    theta = np.full(int(n_features), np.pi / 4, dtype=np.float64)
    cache: dict[tuple[int, ...], tuple[float, dict[str, Any]]] = {}
    best_mask: np.ndarray | None = None
    best_metrics: dict[str, Any] = {}
    best_fitness = float("-inf")
    history: list[dict[str, Any]] = []
    selection_counts = np.zeros(int(n_features), dtype=np.int64)

    for generation in range(1, generations + 1):
        generation_best_mask: np.ndarray | None = None
        generation_best_fitness = float("-inf")
        for chromosome_id in range(1, population_size + 1):
            mask = sample_quantum_mask(
                theta,
                rng,
                min_features=min_features,
                max_features=max_features,
            )
            mask = mutate_mask(
                mask,
                rng,
                mutation_rate=mutation_rate,
                min_features=min_features,
                max_features=max_features,
            )
            key = tuple(int(v) for v in mask.tolist())
            if key not in cache:
                metrics = evaluate_mask(mask, seed + generation * 1000 + chromosome_id)
                fitness = compute_qga_fitness(
                    metrics,
                    features_count=int(mask.sum()),
                    total_features=n_features,
                    weights=weights,
                )
                cache[key] = (float(fitness), dict(metrics))
            fitness, metrics = cache[key]
            selection_counts += mask
            row = {
                "generation": generation,
                "chromosome_id": chromosome_id,
                "fitness": float(fitness),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "attack_recall": float(metrics.get("recall_attack", metrics.get("attack_recall", 0.0))),
                "fpr": float(metrics.get("FPR", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "features_count": int(mask.sum()),
                "selected_indices": " ".join(str(idx) for idx in np.flatnonzero(mask == 1)),
            }
            history.append(row)
            if fitness > generation_best_fitness:
                generation_best_fitness = float(fitness)
                generation_best_mask = mask.copy()
            if fitness > best_fitness:
                best_fitness = float(fitness)
                best_mask = mask.copy()
                best_metrics = dict(metrics)
        assert generation_best_mask is not None
        theta = update_theta_towards_best(theta, generation_best_mask)

    assert best_mask is not None
    probabilities = np.sin(theta) ** 2
    denominator = max(generations * population_size, 1)
    ranking = [
        {
            "feature_index": int(idx),
            "selection_frequency": float(selection_counts[idx] / denominator),
            "theta_probability": float(probabilities[idx]),
            "selected_in_best_mask": bool(best_mask[idx] == 1),
        }
        for idx in range(int(n_features))
    ]
    ranking.sort(key=lambda row: (row["selected_in_best_mask"], row["selection_frequency"], row["theta_probability"]), reverse=True)
    return QGAResult(
        best_mask=best_mask,
        best_metrics=best_metrics,
        best_fitness=float(best_fitness),
        history=history,
        feature_ranking=ranking,
        theta_probabilities=[float(value) for value in probabilities.tolist()],
    )
