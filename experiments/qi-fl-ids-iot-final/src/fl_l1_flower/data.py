"""Data loading for P5.2 Flower L1 clients and server validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fl_l1.client_data import ClientArrays, load_client_npz, make_dataloader
from fl_l1.scenario_loader import L1Scenario, load_l1_scenario, repo_path


@dataclass(frozen=True)
class FlowerClientData:
    """Resolved data for one Flower client."""

    client_id: str
    train: ClientArrays
    val: ClientArrays
    expected_train_samples: int
    expected_val_samples: int


def load_flower_config(config_path: Path) -> dict[str, Any]:
    """Load P5.2 YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def scenario_run_dir(config: dict[str, Any], repo_root: Path, *, alpha: float, clients: int) -> Path:
    """Return P5.2 run directory for one scenario."""

    alpha_name = f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"
    return repo_path(repo_root, config["outputs"]["run_dir"]) / alpha_name / f"k{clients}"


def load_scenario(
    config: dict[str, Any],
    repo_root: Path,
    *,
    alpha: float | None = None,
    clients: int | None = None,
) -> L1Scenario:
    """Load the P3 L1 scenario selected for Flower."""

    resolved_alpha = float(alpha if alpha is not None else config["scenario"]["alpha"])
    resolved_clients = int(clients if clients is not None else config["scenario"]["clients"])
    bridge_config = {
        "inputs": {"partitions_root": config["inputs"]["partitions_root"], "global_test_npz": config["inputs"]["global_test_npz"]},
        "scenario": {
            "default_alpha": resolved_alpha,
            "default_k": resolved_clients,
            "alphas": [resolved_alpha],
            "clients": [resolved_clients],
        },
    }
    return load_l1_scenario(
        bridge_config,
        repo_root,
        alpha=resolved_alpha,
        num_clients=resolved_clients,
    )


def client_id_from_cid(cid: str, num_clients: int) -> str:
    """Map Flower simulation cids to stable P3 client ids."""

    raw = str(cid)
    if raw.startswith("client_"):
        return raw
    try:
        index = int(raw)
    except ValueError:
        if raw.startswith("node"):
            index = int(raw.replace("node", "")) - 1
        else:
            raise ValueError(f"Cannot map Flower cid={cid!r} to client_id") from None
    if index < 0 or index >= int(num_clients):
        raise ValueError(f"Flower cid={cid!r} is outside configured client range 0..{num_clients - 1}")
    return f"client_{index + 1}"


def load_flower_client_data(
    scenario: L1Scenario,
    *,
    client_id: str,
    max_samples: int | None,
    seed: int,
) -> FlowerClientData:
    """Load train/val arrays for one Flower client."""

    partition = next((item for item in scenario.clients if item.client_id == client_id), None)
    if partition is None:
        raise ValueError(f"Unknown client_id={client_id}; expected {[item.client_id for item in scenario.clients]}")
    train = load_client_npz(partition.train_npz, max_samples=max_samples, seed=seed)
    val = load_client_npz(partition.val_npz, max_samples=max_samples, seed=seed + 10_000)
    return FlowerClientData(
        client_id=client_id,
        train=train,
        val=val,
        expected_train_samples=partition.train_samples,
        expected_val_samples=partition.val_samples,
    )


def concatenate_validation_arrays(
    scenario: L1Scenario,
    *,
    max_samples_per_client: int | None,
    seed: int,
) -> ClientArrays:
    """Build the server-side validation union from P3 client validation partitions."""

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    for index, partition in enumerate(scenario.clients, start=1):
        arrays = load_client_npz(
            partition.val_npz,
            max_samples=max_samples_per_client,
            seed=seed + 20_000 + index,
        )
        xs.append(arrays.X)
        ys.append(arrays.y)
        label_ids.append(arrays.label_id_original)
        row_ids.append(arrays.row_id)
    return ClientArrays(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        label_id_original=np.concatenate(label_ids, axis=0),
        row_id=np.concatenate(row_ids, axis=0),
    )


__all__ = [
    "ClientArrays",
    "FlowerClientData",
    "client_id_from_cid",
    "concatenate_validation_arrays",
    "load_flower_client_data",
    "load_flower_config",
    "load_scenario",
    "make_dataloader",
    "repo_path",
    "scenario_run_dir",
]

