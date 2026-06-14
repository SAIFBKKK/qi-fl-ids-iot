"""Data loading for P6 hierarchical L2/L3 Flower experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from fl_l1.scenario_loader import load_json, repo_path
from fl_hierarchical.runtime import alpha_dir


@dataclass(frozen=True)
class HierarchicalArrays:
    X: np.ndarray
    y: np.ndarray
    label_id_original: np.ndarray
    row_id: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class TaskSpec:
    task: str
    short_name: str
    target_key: str
    output_dim: int
    class_names: list[str]
    class_mapping: dict[str, Any]
    model_config: dict[str, Any]


@dataclass(frozen=True)
class ClientPartition:
    client_id: str
    train_row_ids_npy: Path
    val_row_ids_npy: Path
    train_samples: int
    val_samples: int


@dataclass(frozen=True)
class L2IndexScenario:
    alpha: float
    num_clients: int
    scenario_dir: Path
    manifest_path: Path
    distribution_report_path: Path
    global_test_reference_path: Path
    global_test_npz: Path
    clients: list[ClientPartition]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class HierarchicalClientData:
    client_id: str
    train: HierarchicalArrays
    val: HierarchicalArrays
    expected_train_samples: int
    expected_val_samples: int


def load_hierarchical_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def normalize_task(task: str) -> str:
    raw = str(task).lower().strip()
    if raw in {"l2", "l2_family", "family"}:
        return "l2_family"
    if raw in {"l3", "l3_attack_type", "attack_type"}:
        return "l3_attack_type"
    raise ValueError(f"Unsupported P6 task: {task!r}")


def task_short_name(task: str) -> str:
    return "l2" if normalize_task(task) == "l2_family" else "l3"


def load_task_spec(config: dict[str, Any], repo_root: Path, task: str) -> TaskSpec:
    normalized = normalize_task(task)
    short_name = task_short_name(normalized)
    if normalized == "l2_family":
        mapping = load_json(repo_path(repo_root, config["inputs"]["l2_family_mapping"]))
        id_to_family = {int(k): str(v) for k, v in mapping["id_to_family"].items()}
        class_names = [id_to_family[index] for index in sorted(id_to_family)]
        model_cfg = dict(config["models"]["l2"])
        target_key = "y_family"
        class_mapping = {
            "family_to_id": mapping["family_to_id"],
            "id_to_family": mapping["id_to_family"],
        }
    else:
        id_to_label_raw = load_json(repo_path(repo_root, config["inputs"]["id_to_label"]))
        id_to_label = {int(k): str(v) for k, v in id_to_label_raw.items()}
        attack_label_ids = sorted(label_id for label_id, name in id_to_label.items() if name != "BenignTraffic")
        original_to_l3 = {str(label_id): index for index, label_id in enumerate(attack_label_ids)}
        l3_to_original = {str(index): label_id for label_id, index in ((label_id, original_to_l3[str(label_id)]) for label_id in attack_label_ids)}
        class_names = [id_to_label[label_id] for label_id in attack_label_ids]
        model_cfg = dict(config["models"]["l3"])
        target_key = "label_id_original"
        class_mapping = {
            "original_label_id_to_l3_id": original_to_l3,
            "l3_id_to_original_label_id": l3_to_original,
            "l3_id_to_label_name": {str(index): name for index, name in enumerate(class_names)},
            "benign_excluded": True,
        }
    model_cfg["output_dim"] = len(class_names)
    return TaskSpec(
        task=normalized,
        short_name=short_name,
        target_key=target_key,
        output_dim=len(class_names),
        class_names=class_names,
        class_mapping=class_mapping,
        model_config=model_cfg,
    )


def load_l2_index_scenario(
    config: dict[str, Any],
    repo_root: Path,
    *,
    alpha: float | None = None,
    clients: int | None = None,
) -> L2IndexScenario:
    resolved_alpha = float(alpha if alpha is not None else config["scenario"]["default_alpha"])
    resolved_clients = int(clients if clients is not None else config["scenario"]["default_k"])
    scenario_dir = repo_path(repo_root, config["inputs"]["l2_partitions_root"]) / alpha_dir(resolved_alpha) / f"k{resolved_clients}"
    manifest_path = scenario_dir / "manifest.json"
    distribution_report_path = scenario_dir / "distribution_report.json"
    global_test_reference_path = scenario_dir / "global_test_reference.json"
    for path in [scenario_dir, manifest_path, distribution_report_path, global_test_reference_path]:
        if not path.exists():
            raise FileNotFoundError(f"P3 L2 scenario is incomplete, missing: {path}")
    manifest = load_json(manifest_path)
    if manifest.get("dataset_level") != "l2_family":
        raise ValueError(f"expected l2_family scenario, got {manifest.get('dataset_level')!r}")
    if manifest.get("storage_mode") != "index_only":
        raise ValueError("P6 expects P3 L2 index_only partitions")
    if bool(manifest.get("partition_test", True)):
        raise ValueError("P6 refuses scenarios where the global test holdout was partitioned")
    reference = load_json(global_test_reference_path)
    global_test_npz = repo_path(repo_root, reference["global_test_npz"])
    if global_test_npz.resolve() != repo_path(repo_root, config["inputs"]["l2_test_npz"]).resolve():
        raise ValueError("P3 L2 global_test_reference does not match P6 configured holdout")
    row_counts = manifest["row_counts"]["by_client"]
    partitions: list[ClientPartition] = []
    for client_index in range(1, resolved_clients + 1):
        client_id = f"client_{client_index}"
        client_dir = scenario_dir / client_id
        train_path = client_dir / "train_row_ids.npy"
        val_path = client_dir / "val_row_ids.npy"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"missing index_only row ids for {client_id}: {client_dir}")
        if (client_dir / "test_scaled.npz").exists() or (client_dir / "test_row_ids.npy").exists():
            raise ValueError(f"client test partition found and forbidden: {client_dir}")
        partitions.append(
            ClientPartition(
                client_id=client_id,
                train_row_ids_npy=train_path,
                val_row_ids_npy=val_path,
                train_samples=int(row_counts[client_id]["train"]),
                val_samples=int(row_counts[client_id]["val"]),
            )
        )
    return L2IndexScenario(
        alpha=resolved_alpha,
        num_clients=resolved_clients,
        scenario_dir=scenario_dir,
        manifest_path=manifest_path,
        distribution_report_path=distribution_report_path,
        global_test_reference_path=global_test_reference_path,
        global_test_npz=global_test_npz,
        clients=partitions,
        manifest=manifest,
    )


def client_id_from_cid(cid: str, num_clients: int) -> str:
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
        raise ValueError(f"Flower cid={cid!r} outside client range")
    return f"client_{index + 1}"


def _sample_row_ids(path: Path, *, max_samples: int | None, seed: int) -> np.ndarray:
    row_ids = np.load(path, allow_pickle=False)
    if max_samples is not None and row_ids.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(row_ids.shape[0], size=int(max_samples), replace=False)
        row_ids = row_ids[np.sort(chosen)]
    return row_ids.astype(np.int64, copy=False)


def _target_from_npz(npz: Any, task_spec: TaskSpec, indices: np.ndarray) -> np.ndarray:
    if task_spec.task == "l2_family":
        return np.asarray(npz["y_family"][indices], dtype=np.int64)
    label_ids = np.asarray(npz["label_id_original"][indices], dtype=np.int64)
    mapping = {int(k): int(v) for k, v in task_spec.class_mapping["original_label_id_to_l3_id"].items()}
    try:
        return np.asarray([mapping[int(label_id)] for label_id in label_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"L3 received an unknown or benign label_id={exc.args[0]}") from exc


def load_arrays_by_row_ids(
    npz_path: Path,
    row_ids: np.ndarray,
    *,
    task_spec: TaskSpec,
) -> HierarchicalArrays:
    """Load selected rows from a global L2 NPZ using stable row_id values."""

    with np.load(npz_path, allow_pickle=False) as npz:
        global_row_ids = np.asarray(npz["row_id"], dtype=np.int64)
        mask = np.isin(global_row_ids, row_ids, assume_unique=False)
        indices = np.flatnonzero(mask)
        if indices.shape[0] != row_ids.shape[0]:
            raise ValueError(f"row_id selection mismatch for {npz_path}: expected {row_ids.shape[0]}, got {indices.shape[0]}")
        label_ids = np.asarray(npz["label_id_original"][indices], dtype=np.int64)
        if np.any(label_ids == 1):
            raise ValueError("P6 L2/L3 clients must be attack-only; BenignTraffic label_id=1 found")
        return HierarchicalArrays(
            X=np.asarray(npz["X"][indices], dtype=np.float32),
            y=_target_from_npz(npz, task_spec, indices),
            label_id_original=label_ids,
            row_id=np.asarray(global_row_ids[indices], dtype=np.int64),
        )


def load_global_arrays(
    config: dict[str, Any],
    repo_root: Path,
    *,
    split: str,
    task_spec: TaskSpec,
    max_samples: int | None,
    seed: int,
) -> HierarchicalArrays:
    path_key = {"train": "l2_train_npz", "val": "l2_val_npz", "test": "l2_test_npz"}[split]
    npz_path = repo_path(repo_root, config["inputs"][path_key])
    with np.load(npz_path, allow_pickle=False) as npz:
        total = int(npz["row_id"].shape[0])
        if max_samples is not None and total > int(max_samples):
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(total, size=int(max_samples), replace=False))
        else:
            indices = np.arange(total)
        label_ids = np.asarray(npz["label_id_original"][indices], dtype=np.int64)
        if np.any(label_ids == 1):
            raise ValueError("P6 global L2/L3 arrays must be attack-only")
        return HierarchicalArrays(
            X=np.asarray(npz["X"][indices], dtype=np.float32),
            y=_target_from_npz(npz, task_spec, indices),
            label_id_original=label_ids,
            row_id=np.asarray(npz["row_id"][indices], dtype=np.int64),
        )


def load_hierarchical_client_data(
    config: dict[str, Any],
    repo_root: Path,
    scenario: L2IndexScenario,
    task_spec: TaskSpec,
    *,
    client_id: str,
    max_samples_per_client: int | None,
) -> HierarchicalClientData:
    partition = next((item for item in scenario.clients if item.client_id == client_id), None)
    if partition is None:
        raise ValueError(f"Unknown client_id={client_id}")
    seed = int(config["training"]["seed"])
    train_row_ids = _sample_row_ids(
        partition.train_row_ids_npy,
        max_samples=max_samples_per_client,
        seed=seed + int(client_id.split("_")[-1]),
    )
    val_row_ids = _sample_row_ids(
        partition.val_row_ids_npy,
        max_samples=max_samples_per_client,
        seed=seed + 10_000 + int(client_id.split("_")[-1]),
    )
    return HierarchicalClientData(
        client_id=client_id,
        train=load_arrays_by_row_ids(repo_path(repo_root, config["inputs"]["l2_train_npz"]), train_row_ids, task_spec=task_spec),
        val=load_arrays_by_row_ids(repo_path(repo_root, config["inputs"]["l2_val_npz"]), val_row_ids, task_spec=task_spec),
        expected_train_samples=partition.train_samples,
        expected_val_samples=partition.val_samples,
    )


def concatenate_validation_arrays(
    config: dict[str, Any],
    repo_root: Path,
    scenario: L2IndexScenario,
    task_spec: TaskSpec,
    *,
    max_samples_per_client: int | None,
) -> HierarchicalArrays:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    for partition in scenario.clients:
        arrays = load_hierarchical_client_data(
            config,
            repo_root,
            scenario,
            task_spec,
            client_id=partition.client_id,
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


def make_dataloader(
    arrays: HierarchicalArrays,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = TensorDataset(
        torch.from_numpy(arrays.X.astype(np.float32, copy=False)),
        torch.from_numpy(arrays.y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        pin_memory=bool(device is not None and device.type == "cuda"),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
