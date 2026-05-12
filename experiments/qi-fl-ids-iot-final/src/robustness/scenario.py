"""Scenario loading for P10 L1 robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import alpha_dir, resolve


@dataclass(frozen=True)
class ClientPartition:
    client_id: str
    train_npz: Path
    val_npz: Path


@dataclass(frozen=True)
class RobustnessScenario:
    alpha: float
    clients: int
    rounds: int
    attack_type: str
    poison_rate: float
    poisoned_clients: int
    method: str
    partitions: list[ClientPartition]
    test_npz: Path


def load_npz_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "X" not in data or "y_binary" not in data:
            raise ValueError(f"{path} must contain X and y_binary")
        return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y_binary"], dtype=np.int64)


def load_scenario(
    config: dict[str, Any],
    *,
    alpha: float,
    clients: int,
    rounds: int,
    attack_type: str,
    poison_rate: float,
    poisoned_clients: int,
    method: str,
) -> RobustnessScenario:
    partitions_root = resolve(config["inputs"]["l1_partitions_root"])
    scenario_dir = partitions_root / alpha_dir(alpha) / f"k{clients}"
    if not scenario_dir.exists():
        raise FileNotFoundError(f"missing P3 L1 scenario: {scenario_dir}")
    partitions: list[ClientPartition] = []
    for idx in range(1, int(clients) + 1):
        client_dir = scenario_dir / f"client_{idx}"
        train_npz = client_dir / "train_scaled.npz"
        val_npz = client_dir / "val_scaled.npz"
        if not train_npz.exists() or not val_npz.exists():
            raise FileNotFoundError(f"missing train/val npz for {client_dir}")
        partitions.append(ClientPartition(f"client_{idx}", train_npz, val_npz))
    return RobustnessScenario(
        alpha=float(alpha),
        clients=int(clients),
        rounds=int(rounds),
        attack_type=str(attack_type),
        poison_rate=float(poison_rate),
        poisoned_clients=int(poisoned_clients),
        method=str(method),
        partitions=partitions,
        test_npz=resolve(config["inputs"]["l1_test_npz"]),
    )
