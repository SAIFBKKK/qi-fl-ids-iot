"""Data loading for P8-b QGA L2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fl_hierarchical.data import (
    HierarchicalArrays,
    HierarchicalClientData,
    load_global_arrays,
    load_hierarchical_client_data,
    load_l2_index_scenario,
    load_task_spec,
)
from qga_l2.feature_mask import apply_feature_mask


@dataclass(frozen=True)
class MaskedClientData:
    client_id: str
    train: HierarchicalArrays
    val: HierarchicalArrays
    expected_train_samples: int
    expected_val_samples: int


def mask_arrays(arrays: HierarchicalArrays, mask: np.ndarray) -> HierarchicalArrays:
    return HierarchicalArrays(
        X=apply_feature_mask(arrays.X, mask).astype(np.float32, copy=False),
        y=arrays.y,
        label_id_original=arrays.label_id_original,
        row_id=arrays.row_id,
    )


def load_masked_client_data(
    config: dict,
    repo_root: Path,
    *,
    alpha: float,
    clients: int,
    client_id: str,
    mask: np.ndarray,
    max_samples_per_client: int | None,
) -> MaskedClientData:
    task_spec = load_task_spec(config, repo_root, "l2")
    scenario = load_l2_index_scenario(config, repo_root, alpha=alpha, clients=clients)
    raw: HierarchicalClientData = load_hierarchical_client_data(
        config,
        repo_root,
        scenario,
        task_spec,
        client_id=client_id,
        max_samples_per_client=max_samples_per_client,
    )
    return MaskedClientData(
        client_id=client_id,
        train=mask_arrays(raw.train, mask),
        val=mask_arrays(raw.val, mask),
        expected_train_samples=raw.expected_train_samples,
        expected_val_samples=raw.expected_val_samples,
    )


def concatenate_masked_validation_arrays(
    config: dict,
    repo_root: Path,
    *,
    alpha: float,
    clients: int,
    mask: np.ndarray,
    max_samples_per_client: int | None,
) -> HierarchicalArrays:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    for client_index in range(1, int(clients) + 1):
        arrays = load_masked_client_data(
            config,
            repo_root,
            alpha=alpha,
            clients=clients,
            client_id=f"client_{client_index}",
            mask=mask,
            max_samples_per_client=max_samples_per_client,
        ).val
        xs.append(arrays.X)
        ys.append(arrays.y)
        label_ids.append(arrays.label_id_original)
        row_ids.append(arrays.row_id)
    return HierarchicalArrays(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        label_id_original=np.concatenate(label_ids, axis=0),
        row_id=np.concatenate(row_ids, axis=0),
    )


def load_masked_global_arrays(
    config: dict,
    repo_root: Path,
    *,
    split: str,
    mask: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> HierarchicalArrays:
    task_spec = load_task_spec(config, repo_root, "l2")
    return mask_arrays(
        load_global_arrays(config, repo_root, split=split, task_spec=task_spec, max_samples=max_samples, seed=seed),
        mask,
    )
