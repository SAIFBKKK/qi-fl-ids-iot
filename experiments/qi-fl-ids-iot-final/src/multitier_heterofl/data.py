"""Data loading for P7 L1/L2 HeteroFL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fl_l1.client_data import ClientArrays, load_client_npz
from fl_l1.scenario_loader import load_l1_scenario
from fl_hierarchical.data import (
    HierarchicalArrays,
    concatenate_validation_arrays as concat_l2_validation,
    load_global_arrays as load_l2_global_arrays,
    load_hierarchical_client_data,
    load_l2_index_scenario,
    load_task_spec,
)
from multitier_heterofl.config import normalize_task, repo_path


@dataclass(frozen=True)
class P7TaskSpec:
    task: str
    output_dim: int
    class_names: list[str]
    target_name: str


@dataclass(frozen=True)
class P7ClientData:
    client_id: str
    tier: str
    train: ClientArrays | HierarchicalArrays
    val: ClientArrays | HierarchicalArrays
    expected_train_samples: int
    expected_val_samples: int


@dataclass(frozen=True)
class P7Scenario:
    task: str
    alpha: float
    num_clients: int
    scenario_dir: Path
    global_test_npz: Path
    client_ids: list[str]
    client_train_counts: dict[str, int]
    client_val_counts: dict[str, int]
    raw: Any


def task_spec(config: dict[str, Any], repo_root: Path, task: str) -> P7TaskSpec:
    normalized = normalize_task(task)
    if normalized == "l1_binary":
        return P7TaskSpec(task=normalized, output_dim=2, class_names=["normal", "attack"], target_name="y_binary")
    l2_spec = load_task_spec(
        {
            "inputs": {
                "l2_family_mapping": config["inputs"]["l2_family_mapping"],
                "id_to_label": config["inputs"].get("id_to_label", ""),
            },
            "models": {"l2": {"input_dim": 28, "hidden_layers": [128, 64], "dropout": 0.2, "activation": "relu"}},
        },
        repo_root,
        "l2",
    )
    return P7TaskSpec(task=normalized, output_dim=l2_spec.output_dim, class_names=l2_spec.class_names, target_name="y_family")


def load_scenario(config: dict[str, Any], repo_root: Path, *, task: str, alpha: float, clients: int) -> P7Scenario:
    normalized = normalize_task(task)
    if normalized == "l1_binary":
        bridge = {
            "inputs": {
                "partitions_root": config["inputs"]["l1_partitions_root"],
                "global_test_npz": config["inputs"]["l1_test_npz"],
            },
            "scenario": {"default_alpha": alpha, "default_k": clients},
        }
        scenario = load_l1_scenario(bridge, repo_root, alpha=alpha, num_clients=clients)
        return P7Scenario(
            task=normalized,
            alpha=float(alpha),
            num_clients=int(clients),
            scenario_dir=scenario.scenario_dir,
            global_test_npz=scenario.global_test_npz,
            client_ids=[client.client_id for client in scenario.clients],
            client_train_counts={client.client_id: int(client.train_samples) for client in scenario.clients},
            client_val_counts={client.client_id: int(client.val_samples) for client in scenario.clients},
            raw=scenario,
        )
    bridge = {
        "inputs": {
            "l2_partitions_root": config["inputs"]["l2_partitions_root"],
            "l2_test_npz": config["inputs"]["l2_test_npz"],
        }
    }
    scenario = load_l2_index_scenario(bridge, repo_root, alpha=alpha, clients=clients)
    return P7Scenario(
        task=normalized,
        alpha=float(alpha),
        num_clients=int(clients),
        scenario_dir=scenario.scenario_dir,
        global_test_npz=scenario.global_test_npz,
        client_ids=[client.client_id for client in scenario.clients],
        client_train_counts={client.client_id: int(client.train_samples) for client in scenario.clients},
        client_val_counts={client.client_id: int(client.val_samples) for client in scenario.clients},
        raw=scenario,
    )


def load_client_data(
    config: dict[str, Any],
    repo_root: Path,
    scenario: P7Scenario,
    spec: P7TaskSpec,
    *,
    client_id: str,
    tier: str,
    max_samples: int | None,
) -> P7ClientData:
    seed = int(config["training"]["seed"])
    if scenario.task == "l1_binary":
        partition = next(item for item in scenario.raw.clients if item.client_id == client_id)
        train = load_client_npz(partition.train_npz, max_samples=max_samples, seed=seed + int(client_id.split("_")[-1]))
        val = load_client_npz(partition.val_npz, max_samples=max_samples, seed=seed + 10_000 + int(client_id.split("_")[-1]))
        return P7ClientData(client_id, tier, train, val, partition.train_samples, partition.val_samples)
    l2_spec = load_task_spec(
        {
            "inputs": {
                "l2_family_mapping": config["inputs"]["l2_family_mapping"],
                "id_to_label": config["inputs"].get("id_to_label", ""),
            },
            "models": {"l2": {"input_dim": 28, "hidden_layers": [128, 64], "dropout": 0.2, "activation": "relu"}},
        },
        repo_root,
        "l2",
    )
    data = load_hierarchical_client_data(
        config,
        repo_root,
        scenario.raw,
        l2_spec,
        client_id=client_id,
        max_samples_per_client=max_samples,
    )
    return P7ClientData(client_id, tier, data.train, data.val, data.expected_train_samples, data.expected_val_samples)


def load_validation_union(
    config: dict[str, Any],
    repo_root: Path,
    scenario: P7Scenario,
    spec: P7TaskSpec,
    *,
    max_samples_per_client: int | None,
) -> ClientArrays | HierarchicalArrays:
    if scenario.task == "l1_binary":
        xs = []
        ys = []
        labels = []
        rows = []
        for partition in scenario.raw.clients:
            arrays = load_client_npz(
                partition.val_npz,
                max_samples=max_samples_per_client,
                seed=int(config["training"]["seed"]) + 20_000 + int(partition.client_id.split("_")[-1]),
            )
            xs.append(arrays.X)
            ys.append(arrays.y)
            labels.append(arrays.label_id_original)
            rows.append(arrays.row_id)
        return ClientArrays(
            X=np.concatenate(xs, axis=0),
            y=np.concatenate(ys, axis=0),
            label_id_original=np.concatenate(labels, axis=0),
            row_id=np.concatenate(rows, axis=0),
        )
    l2_spec = load_task_spec(
        {
            "inputs": {
                "l2_family_mapping": config["inputs"]["l2_family_mapping"],
                "id_to_label": config["inputs"].get("id_to_label", ""),
            },
            "models": {"l2": {"input_dim": 28, "hidden_layers": [128, 64], "dropout": 0.2, "activation": "relu"}},
        },
        repo_root,
        "l2",
    )
    return concat_l2_validation(config, repo_root, scenario.raw, l2_spec, max_samples_per_client=max_samples_per_client)


def load_global_test(
    config: dict[str, Any],
    repo_root: Path,
    spec: P7TaskSpec,
    *,
    max_samples: int | None,
) -> ClientArrays | HierarchicalArrays:
    if spec.task == "l1_binary":
        return load_client_npz(repo_path(repo_root, config["inputs"]["l1_test_npz"]), max_samples=max_samples, seed=int(config["training"]["seed"]) + 99_000)
    l2_spec = load_task_spec(
        {
            "inputs": {
                "l2_family_mapping": config["inputs"]["l2_family_mapping"],
                "id_to_label": config["inputs"].get("id_to_label", ""),
            },
            "models": {"l2": {"input_dim": 28, "hidden_layers": [128, 64], "dropout": 0.2, "activation": "relu"}},
        },
        repo_root,
        "l2",
    )
    return load_l2_global_arrays(
        config,
        repo_root,
        split="test",
        task_spec=l2_spec,
        max_samples=max_samples,
        seed=int(config["training"]["seed"]) + 99_000,
    )
