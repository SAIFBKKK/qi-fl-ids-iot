"""P3 Dirichlet non-IID partitioning for L1 and L2 datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class DirichletPartition:
    """Client index allocation returned by the Dirichlet splitter."""

    client_indices: list[np.ndarray]
    attempts_used: int
    warnings: list[str]
    proportions_by_class: dict[str, list[float]]


@dataclass(frozen=True)
class DirichletRun:
    """Result returned by the P3 pipeline."""

    summary: dict[str, Any]
    profile: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    generated_files: list[str]

    @property
    def accepted(self) -> bool:
        return not self.errors and bool(self.summary.get("accepted", False))


def load_config(config_path: Path) -> dict[str, Any]:
    """Load P3 YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def _repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def _rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def _label_name_map(raw: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in raw.items()}


def _file_info(path: Path, repo_root: Path) -> dict[str, Any]:
    return {"path": _rel(path, repo_root), "size_bytes": int(path.stat().st_size)}


def _integer_partition_sizes(
    proportions: np.ndarray, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    raw_sizes = proportions * n_items
    sizes = np.floor(raw_sizes).astype(np.int64)
    remainder = int(n_items - sizes.sum())
    if remainder > 0:
        fractional = raw_sizes - sizes
        order = np.argsort(-fractional)
        ties = np.flatnonzero(fractional == fractional[order[0]])
        if ties.size > 1:
            rng.shuffle(order)
        sizes[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(-sizes)
        for idx in order[: abs(remainder)]:
            if sizes[idx] > 0:
                sizes[idx] -= 1
    if int(sizes.sum()) != n_items:
        sizes[-1] += n_items - int(sizes.sum())
    return sizes


def _build_partition_once(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    client_parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    proportions_by_class: dict[str, list[float]] = {}

    for class_id in sorted(np.unique(y).tolist()):
        class_indices = np.flatnonzero(y == class_id).astype(np.int64)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha), dtype=np.float64))
        sizes = _integer_partition_sizes(proportions, class_indices.size, rng)
        proportions_by_class[str(int(class_id))] = [float(value) for value in proportions]

        cursor = 0
        for client_idx, size in enumerate(sizes.tolist()):
            client_parts[client_idx].append(class_indices[cursor : cursor + size])
            cursor += size

    client_indices = []
    for parts in client_parts:
        if parts:
            client_indices.append(np.sort(np.concatenate(parts).astype(np.int64)))
        else:
            client_indices.append(np.empty(0, dtype=np.int64))
    return client_indices, proportions_by_class


def _repair_empty_clients(client_indices: list[np.ndarray]) -> list[np.ndarray]:
    repaired = [np.array(indices, copy=True) for indices in client_indices]
    empty_clients = [idx for idx, indices in enumerate(repaired) if indices.size == 0]
    for empty_idx in empty_clients:
        donor_idx = int(np.argmax([indices.size for indices in repaired]))
        if repaired[donor_idx].size <= 1:
            break
        moved = repaired[donor_idx][-1:]
        repaired[donor_idx] = repaired[donor_idx][:-1]
        repaired[empty_idx] = moved
    return [np.sort(indices.astype(np.int64)) for indices in repaired]


def dirichlet_partition_indices(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int,
    max_attempts: int,
) -> DirichletPartition:
    """Partition local sample indices by class using a Dirichlet draw."""

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if y.size == 0:
        raise ValueError("cannot partition an empty target array")

    best: tuple[list[np.ndarray], dict[str, list[float]], int] | None = None
    best_min_count = -1

    for attempt in range(1, max_attempts + 1):
        client_indices, proportions_by_class = _build_partition_once(
            y, num_clients, alpha, seed + attempt - 1
        )
        counts = [indices.size for indices in client_indices]
        min_count = min(counts)
        if min_count > best_min_count:
            best = (client_indices, proportions_by_class, attempt)
            best_min_count = min_count
        if min_count >= min_samples_per_client:
            return DirichletPartition(
                client_indices=client_indices,
                attempts_used=attempt,
                warnings=[],
                proportions_by_class=proportions_by_class,
            )

    if best is None:
        raise RuntimeError("Dirichlet partitioning did not produce any attempt")

    client_indices, proportions_by_class, attempts_used = best
    warnings: list[str] = [
        f"min_samples_per_client={min_samples_per_client} was not met after "
        f"{max_attempts} attempts; smallest client has {best_min_count} rows."
    ]
    if best_min_count == 0:
        client_indices = _repair_empty_clients(client_indices)
        repaired_min = min(indices.size for indices in client_indices)
        warnings.append(
            "empty clients were repaired by moving one sample from the largest client"
        )
        if repaired_min == 0:
            raise RuntimeError("Dirichlet partitioning produced an empty client")

    return DirichletPartition(
        client_indices=client_indices,
        attempts_used=attempts_used,
        warnings=warnings,
        proportions_by_class=proportions_by_class,
    )


def _validate_partition_indices(client_indices: list[np.ndarray], total_rows: int) -> dict[str, Any]:
    lengths = [int(indices.size) for indices in client_indices]
    if lengths:
        concatenated = np.concatenate(client_indices)
    else:
        concatenated = np.empty(0, dtype=np.int64)
    unique = np.unique(concatenated)
    return {
        "total_assigned": int(concatenated.size),
        "unique_assigned": int(unique.size),
        "expected_total": int(total_rows),
        "no_missing_indices": int(concatenated.size) == int(total_rows),
        "no_duplicate_indices": int(unique.size) == int(concatenated.size),
        "client_counts": {
            f"client_{idx + 1}": count for idx, count in enumerate(lengths)
        },
        "empty_clients": [
            f"client_{idx + 1}" for idx, count in enumerate(lengths) if count == 0
        ],
    }


def _load_l1_npz(path: Path, target_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"].astype(np.float32, copy=False),
            target_key: data[target_key].astype(np.int16, copy=False),
            "label_id_original": data["label_id_original"].astype(np.int16, copy=False),
            "row_id": data["row_id"].astype(np.int64, copy=False),
        }


def _load_index_npz(path: Path, target_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            target_key: data[target_key].astype(np.int16, copy=False),
            "label_id_original": data["label_id_original"].astype(np.int16, copy=False),
            "row_id": data["row_id"].astype(np.int64, copy=False),
        }


def _save_l1_client_npz(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    output_path: Path,
    target_key: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X=data["X"][indices],
        y_binary=data[target_key][indices],
        label_id_original=data["label_id_original"][indices],
        row_id=data["row_id"][indices],
    )


def _counts_for_indices(
    values: np.ndarray,
    indices: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, dict[str, Any]]:
    selected = values[indices]
    total = int(selected.size)
    counts: dict[str, dict[str, Any]] = {}
    for class_id in sorted(class_names):
        count = int(np.sum(selected == class_id))
        counts[str(class_id)] = {
            "class_name": class_names[class_id],
            "count": count,
            "ratio_within_client": count / total if total else 0.0,
        }
    return counts


def _label_counts_for_indices(
    label_ids: np.ndarray,
    indices: np.ndarray,
) -> dict[str, int]:
    selected = label_ids[indices]
    values, counts = np.unique(selected, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _client_distribution_rows(
    dataset_level: str,
    alpha: float,
    num_clients: int,
    split: str,
    target_values: np.ndarray,
    client_indices: list[np.ndarray],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for client_idx, indices in enumerate(client_indices, start=1):
        counts = _counts_for_indices(target_values, indices, class_names)
        for class_id, payload in counts.items():
            rows.append(
                {
                    "dataset_level": dataset_level,
                    "alpha": alpha,
                    "num_clients": num_clients,
                    "split": split,
                    "client_id": f"client_{client_idx}",
                    "class_id": int(class_id),
                    "class_name": payload["class_name"],
                    "count": payload["count"],
                    "ratio_within_client": payload["ratio_within_client"],
                }
            )
    return rows


def _write_client_distribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_level",
        "alpha",
        "num_clients",
        "split",
        "client_id",
        "class_id",
        "class_name",
        "count",
        "ratio_within_client",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _anti_leakage_result(
    train_row_ids: list[np.ndarray],
    val_row_ids: list[np.ndarray],
    *,
    expected_train: int,
    expected_val: int,
) -> dict[str, Any]:
    train_all = np.concatenate(train_row_ids) if train_row_ids else np.empty(0, dtype=np.int64)
    val_all = np.concatenate(val_row_ids) if val_row_ids else np.empty(0, dtype=np.int64)
    train_unique = np.unique(train_all)
    val_unique = np.unique(val_all)
    train_val_overlap = int(np.intersect1d(train_unique, val_unique, assume_unique=True).size)
    return {
        "anti_leakage_id": "row_id",
        "train_union_matches_global_count": int(train_all.size) == int(expected_train),
        "val_union_matches_global_count": int(val_all.size) == int(expected_val),
        "train_no_duplicate_row_ids": int(train_unique.size) == int(train_all.size),
        "val_no_duplicate_row_ids": int(val_unique.size) == int(val_all.size),
        "train_val_overlap": train_val_overlap,
        "no_overlap_between_clients": int(train_unique.size) == int(train_all.size)
        and int(val_unique.size) == int(val_all.size),
        "no_train_val_leakage": train_val_overlap == 0,
        "valid": int(train_all.size) == int(expected_train)
        and int(val_all.size) == int(expected_val)
        and int(train_unique.size) == int(train_all.size)
        and int(val_unique.size) == int(val_all.size)
        and train_val_overlap == 0,
    }


def _global_test_reference(
    path: Path,
    repo_root: Path,
    dataset_level: str,
) -> dict[str, Any]:
    return {
        "dataset_level": dataset_level,
        "partition_test": False,
        "keep_global_test_holdout": True,
        "global_test_npz": _rel(path, repo_root),
        "size_bytes": int(path.stat().st_size),
        "intended_uses": [
            "final global model evaluation",
            "offline final validation",
            "microservices deployment tests",
            "dashboard and inference demo simulation",
        ],
    }


def _write_global_test_reference(
    path: Path,
    reference: dict[str, Any],
) -> None:
    _write_json(path, reference)


def _scenario_dir(base: Path, dataset_level: str, alpha: float, num_clients: int) -> Path:
    return base / dataset_level / _alpha_dir(alpha) / f"k{num_clients}"


def _build_manifest(
    *,
    repo_root: Path,
    scenario_dir: Path,
    dataset_level: str,
    alpha: float,
    num_clients: int,
    seed: int,
    source_train_npz: Path,
    source_val_npz: Path,
    source_test_npz: Path,
    storage_mode: str,
    train_partition: DirichletPartition,
    val_partition: DirichletPartition,
    train_row_ids: list[np.ndarray],
    val_row_ids: list[np.ndarray],
    distribution_report: dict[str, Any],
    anti_leakage: dict[str, Any],
    global_test_reference_file: Path,
    client_files: dict[str, dict[str, str]],
    output_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "created_at": _created_at(),
        "dataset_level": dataset_level,
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "source_train_npz": _rel(source_train_npz, repo_root),
        "source_val_npz": _rel(source_val_npz, repo_root),
        "source_global_test_npz": _rel(source_test_npz, repo_root),
        "partition_train": True,
        "partition_val": True,
        "partition_test": False,
        "keep_global_test_holdout": True,
        "storage_mode": storage_mode,
        "client_ids": [f"client_{idx + 1}" for idx in range(num_clients)],
        "client_files": client_files,
        "row_counts": {
            "train_total": int(sum(ids.size for ids in train_row_ids)),
            "val_total": int(sum(ids.size for ids in val_row_ids)),
            "by_client": {
                f"client_{idx + 1}": {
                    "train": int(train_row_ids[idx].size),
                    "val": int(val_row_ids[idx].size),
                }
                for idx in range(num_clients)
            },
        },
        "class_distributions": distribution_report["class_distributions"],
        "label_id_original_distributions": distribution_report[
            "label_id_original_distributions"
        ],
        "resampling_attempts": {
            "train": train_partition.attempts_used,
            "val": val_partition.attempts_used,
        },
        "warnings": distribution_report["warnings"],
        "anti_leakage_result": anti_leakage,
        "global_test_reference_file": _rel(global_test_reference_file, repo_root),
        "output_files": output_files,
        "scenario_dir": _rel(scenario_dir, repo_root),
    }


def _write_l1_scenario(
    *,
    repo_root: Path,
    scenario_dir: Path,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    train_partition: DirichletPartition,
    val_partition: DirichletPartition,
    alpha: float,
    num_clients: int,
    seed: int,
    source_train_npz: Path,
    source_val_npz: Path,
    source_test_npz: Path,
    class_names: dict[int, str],
) -> dict[str, Any]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    client_files: dict[str, dict[str, str]] = {}
    train_row_ids: list[np.ndarray] = []
    val_row_ids: list[np.ndarray] = []

    for client_idx in range(num_clients):
        client_id = f"client_{client_idx + 1}"
        client_dir = scenario_dir / client_id
        train_path = client_dir / "train_scaled.npz"
        val_path = client_dir / "val_scaled.npz"
        train_indices = train_partition.client_indices[client_idx]
        val_indices = val_partition.client_indices[client_idx]
        _save_l1_client_npz(train_data, train_indices, train_path, "y_binary")
        _save_l1_client_npz(val_data, val_indices, val_path, "y_binary")
        train_row_ids.append(train_data["row_id"][train_indices])
        val_row_ids.append(val_data["row_id"][val_indices])
        client_files[client_id] = {
            "train_scaled_npz": _rel(train_path, repo_root),
            "val_scaled_npz": _rel(val_path, repo_root),
        }

    distribution_rows = _client_distribution_rows(
        "l1_binary",
        alpha,
        num_clients,
        "train",
        train_data["y_binary"],
        train_partition.client_indices,
        class_names,
    )
    distribution_rows.extend(
        _client_distribution_rows(
            "l1_binary",
            alpha,
            num_clients,
            "val",
            val_data["y_binary"],
            val_partition.client_indices,
            class_names,
        )
    )

    class_distributions = {
        "train": {
            f"client_{idx + 1}": _counts_for_indices(
                train_data["y_binary"], indices, class_names
            )
            for idx, indices in enumerate(train_partition.client_indices)
        },
        "val": {
            f"client_{idx + 1}": _counts_for_indices(
                val_data["y_binary"], indices, class_names
            )
            for idx, indices in enumerate(val_partition.client_indices)
        },
    }
    label_distributions = {
        "train": {
            f"client_{idx + 1}": _label_counts_for_indices(
                train_data["label_id_original"], indices
            )
            for idx, indices in enumerate(train_partition.client_indices)
        },
        "val": {
            f"client_{idx + 1}": _label_counts_for_indices(
                val_data["label_id_original"], indices
            )
            for idx, indices in enumerate(val_partition.client_indices)
        },
    }

    anti_leakage = _anti_leakage_result(
        train_row_ids,
        val_row_ids,
        expected_train=train_data["row_id"].size,
        expected_val=val_data["row_id"].size,
    )
    warnings = [*train_partition.warnings, *val_partition.warnings]
    distribution_report = {
        "dataset_level": "l1_binary",
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "total_train_rows": int(train_data["row_id"].size),
        "total_val_rows": int(val_data["row_id"].size),
        "global_test_holdout_path": _rel(source_test_npz, repo_root),
        "partition_test": False,
        "keep_global_test_holdout": True,
        "resampling_attempts": {
            "train": train_partition.attempts_used,
            "val": val_partition.attempts_used,
        },
        "warnings": warnings,
        "class_distributions": class_distributions,
        "label_id_original_distributions": label_distributions,
        "ratios_by_client": class_distributions,
    }

    distribution_report_path = scenario_dir / "distribution_report.json"
    _write_json(distribution_report_path, distribution_report)

    csv_path = scenario_dir / "client_distribution.csv"
    _write_client_distribution_csv(csv_path, distribution_rows)

    reference_path = scenario_dir / "global_test_reference.json"
    _write_global_test_reference(
        reference_path,
        _global_test_reference(source_test_npz, repo_root, "l1_binary"),
    )

    output_files = {
        "manifest": _rel(scenario_dir / "manifest.json", repo_root),
        "distribution_report": _rel(distribution_report_path, repo_root),
        "client_distribution_csv": _rel(csv_path, repo_root),
        "global_test_reference": _rel(reference_path, repo_root),
    }
    manifest = _build_manifest(
        repo_root=repo_root,
        scenario_dir=scenario_dir,
        dataset_level="l1_binary",
        alpha=alpha,
        num_clients=num_clients,
        seed=seed,
        source_train_npz=source_train_npz,
        source_val_npz=source_val_npz,
        source_test_npz=source_test_npz,
        storage_mode="materialized_npz",
        train_partition=train_partition,
        val_partition=val_partition,
        train_row_ids=train_row_ids,
        val_row_ids=val_row_ids,
        distribution_report=distribution_report,
        anti_leakage=anti_leakage,
        global_test_reference_file=reference_path,
        client_files=client_files,
        output_files=output_files,
    )
    manifest_path = scenario_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "distribution_report": distribution_report,
        "files": [manifest_path, distribution_report_path, csv_path, reference_path],
    }


def _write_l2_scenario(
    *,
    repo_root: Path,
    scenario_dir: Path,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    train_partition: DirichletPartition,
    val_partition: DirichletPartition,
    alpha: float,
    num_clients: int,
    seed: int,
    source_train_npz: Path,
    source_val_npz: Path,
    source_test_npz: Path,
    class_names: dict[int, str],
) -> dict[str, Any]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    client_files: dict[str, dict[str, str]] = {}
    train_row_ids: list[np.ndarray] = []
    val_row_ids: list[np.ndarray] = []

    for client_idx in range(num_clients):
        client_id = f"client_{client_idx + 1}"
        client_dir = scenario_dir / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        train_indices = train_partition.client_indices[client_idx]
        val_indices = val_partition.client_indices[client_idx]
        train_ids = np.sort(train_data["row_id"][train_indices].astype(np.int64))
        val_ids = np.sort(val_data["row_id"][val_indices].astype(np.int64))
        train_path = client_dir / "train_row_ids.npy"
        val_path = client_dir / "val_row_ids.npy"
        np.save(train_path, train_ids)
        np.save(val_path, val_ids)
        train_row_ids.append(train_ids)
        val_row_ids.append(val_ids)
        client_files[client_id] = {
            "train_row_ids_npy": _rel(train_path, repo_root),
            "val_row_ids_npy": _rel(val_path, repo_root),
        }

    distribution_rows = _client_distribution_rows(
        "l2_family",
        alpha,
        num_clients,
        "train",
        train_data["y_family"],
        train_partition.client_indices,
        class_names,
    )
    distribution_rows.extend(
        _client_distribution_rows(
            "l2_family",
            alpha,
            num_clients,
            "val",
            val_data["y_family"],
            val_partition.client_indices,
            class_names,
        )
    )

    class_distributions = {
        "train": {
            f"client_{idx + 1}": _counts_for_indices(
                train_data["y_family"], indices, class_names
            )
            for idx, indices in enumerate(train_partition.client_indices)
        },
        "val": {
            f"client_{idx + 1}": _counts_for_indices(
                val_data["y_family"], indices, class_names
            )
            for idx, indices in enumerate(val_partition.client_indices)
        },
    }
    label_distributions = {
        "train": {
            f"client_{idx + 1}": _label_counts_for_indices(
                train_data["label_id_original"], indices
            )
            for idx, indices in enumerate(train_partition.client_indices)
        },
        "val": {
            f"client_{idx + 1}": _label_counts_for_indices(
                val_data["label_id_original"], indices
            )
            for idx, indices in enumerate(val_partition.client_indices)
        },
    }

    anti_leakage = _anti_leakage_result(
        train_row_ids,
        val_row_ids,
        expected_train=train_data["row_id"].size,
        expected_val=val_data["row_id"].size,
    )
    warnings = [*train_partition.warnings, *val_partition.warnings]
    distribution_report = {
        "dataset_level": "l2_family",
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "total_train_rows": int(train_data["row_id"].size),
        "total_val_rows": int(val_data["row_id"].size),
        "global_test_holdout_path": _rel(source_test_npz, repo_root),
        "partition_test": False,
        "keep_global_test_holdout": True,
        "resampling_attempts": {
            "train": train_partition.attempts_used,
            "val": val_partition.attempts_used,
        },
        "warnings": warnings,
        "class_distributions": class_distributions,
        "label_id_original_distributions": label_distributions,
        "ratios_by_client": class_distributions,
    }

    distribution_report_path = scenario_dir / "distribution_report.json"
    _write_json(distribution_report_path, distribution_report)

    csv_path = scenario_dir / "client_distribution.csv"
    _write_client_distribution_csv(csv_path, distribution_rows)

    reference_path = scenario_dir / "global_test_reference.json"
    _write_global_test_reference(
        reference_path,
        _global_test_reference(source_test_npz, repo_root, "l2_family"),
    )

    output_files = {
        "manifest": _rel(scenario_dir / "manifest.json", repo_root),
        "distribution_report": _rel(distribution_report_path, repo_root),
        "client_distribution_csv": _rel(csv_path, repo_root),
        "global_test_reference": _rel(reference_path, repo_root),
    }
    manifest = _build_manifest(
        repo_root=repo_root,
        scenario_dir=scenario_dir,
        dataset_level="l2_family",
        alpha=alpha,
        num_clients=num_clients,
        seed=seed,
        source_train_npz=source_train_npz,
        source_val_npz=source_val_npz,
        source_test_npz=source_test_npz,
        storage_mode="index_only",
        train_partition=train_partition,
        val_partition=val_partition,
        train_row_ids=train_row_ids,
        val_row_ids=val_row_ids,
        distribution_report=distribution_report,
        anti_leakage=anti_leakage,
        global_test_reference_file=reference_path,
        client_files=client_files,
        output_files=output_files,
    )
    manifest_path = scenario_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "distribution_report": distribution_report,
        "files": [manifest_path, distribution_report_path, csv_path, reference_path],
    }


def _verify_p2(repo_root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    final_dir = _repo_path(repo_root, config["final_experiment_dir"])
    required_paths = {
        "p2_report": final_dir / "docs" / "02_preprocessing.md",
        "l1_train_npz": _repo_path(repo_root, config["inputs"]["l1"]["train_npz"]),
        "l1_val_npz": _repo_path(repo_root, config["inputs"]["l1"]["val_npz"]),
        "l1_test_npz": _repo_path(repo_root, config["inputs"]["l1"]["test_npz"]),
        "l1_manifest": _repo_path(repo_root, config["inputs"]["l1"]["manifest"]),
        "l2_train_npz": _repo_path(repo_root, config["inputs"]["l2"]["train_npz"]),
        "l2_val_npz": _repo_path(repo_root, config["inputs"]["l2"]["val_npz"]),
        "l2_test_npz": _repo_path(repo_root, config["inputs"]["l2"]["test_npz"]),
        "l2_manifest": _repo_path(repo_root, config["inputs"]["l2"]["manifest"]),
    }
    for name, path in required_paths.items():
        if not path.exists():
            errors.append(f"missing required P2 artifact: {name} at {_rel(path, repo_root)}")

    p2_summary_path = final_dir / "outputs" / "reports" / "preprocessing_summary.json"
    p2_accepted = True
    if p2_summary_path.exists():
        p2_summary = _load_json(p2_summary_path)
        p2_accepted = bool(p2_summary.get("accepted", False))
        if not p2_accepted:
            errors.append("P2 summary exists but is not accepted")
    else:
        warnings.append("P2 summary JSON not found; relying on docs/02_preprocessing.md")

    if required_paths["p2_report"].exists():
        text = required_paths["p2_report"].read_text(encoding="utf-8")
        if "P2 est validée" not in text and "P2 est validee" not in text:
            warnings.append("P2 markdown report does not contain the expected validation sentence")

    return {
        "required_paths": {
            name: _rel(path, repo_root) for name, path in required_paths.items()
        },
        "p2_summary_path": _rel(p2_summary_path, repo_root)
        if p2_summary_path.exists()
        else None,
        "p2_accepted": p2_accepted,
    }, errors, warnings


def _figure_samples_per_client(
    scenarios: list[dict[str, Any]],
    dataset_level: str,
    path: Path,
) -> None:
    selected = [item for item in scenarios if item["dataset_level"] == dataset_level]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)
    for ax, scenario in zip(axes.ravel(), selected):
        row_counts = scenario["manifest"]["row_counts"]["by_client"]
        labels = list(row_counts.keys())
        train_counts = [row_counts[label]["train"] for label in labels]
        val_counts = [row_counts[label]["val"] for label in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, train_counts, 0.36, label="train", color="#2563EB")
        ax.bar(x + 0.18, val_counts, 0.36, label="val", color="#16A34A")
        ax.set_title(f"alpha={scenario['alpha']}, K={scenario['num_clients']}")
        ax.set_xticks(x, labels=[label.replace("client_", "c") for label in labels])
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].legend()
    fig.suptitle(f"{dataset_level} samples per client")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_heatmaps(
    scenarios: list[dict[str, Any]],
    dataset_level: str,
    class_names: dict[int, str],
    path: Path,
) -> None:
    selected = [item for item in scenarios if item["dataset_level"] == dataset_level]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    class_ids = sorted(class_names)
    for ax, scenario in zip(axes.ravel(), selected):
        distributions = scenario["manifest"]["class_distributions"]["train"]
        matrix = []
        y_labels = []
        for client_id, counts in distributions.items():
            y_labels.append(client_id.replace("client_", "c"))
            matrix.append([counts[str(class_id)]["ratio_within_client"] for class_id in class_ids])
        image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"alpha={scenario['alpha']}, K={scenario['num_clients']}")
        ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
        ax.set_xticks(
            np.arange(len(class_ids)),
            labels=[class_names[class_id] for class_id in class_ids],
            rotation=55,
            ha="right",
            fontsize=7,
        )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="ratio within client")
    fig.suptitle(f"{dataset_level} train class-ratio heatmaps")
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.90, wspace=0.32, hspace=0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_alpha_comparison(
    scenarios: list[dict[str, Any]],
    dataset_level: str,
    path: Path,
) -> None:
    selected = [item for item in scenarios if item["dataset_level"] == dataset_level]
    values: dict[float, list[float]] = {}
    for scenario in selected:
        distributions = scenario["manifest"]["class_distributions"]["train"]
        client_totals = []
        for counts in distributions.values():
            ratios = [payload["ratio_within_client"] for payload in counts.values()]
            client_totals.append(float(np.std(ratios)))
        values.setdefault(float(scenario["alpha"]), []).append(float(np.mean(client_totals)))

    alphas = sorted(values)
    means = [float(np.mean(values[alpha])) for alpha in alphas]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([str(alpha) for alpha in alphas], means, marker="o", color="#DC2626")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean client class-ratio std")
    ax.set_title(f"{dataset_level} non-IID strength by alpha")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_global_test_holdout(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        ("P2 train", 0.12, 0.68, "#DBEAFE"),
        ("P2 val", 0.12, 0.35, "#DCFCE7"),
        ("P2 global test", 0.12, 0.12, "#FEF3C7"),
        ("Dirichlet clients\ntrain/val only", 0.58, 0.52, "#E0E7FF"),
        ("Holdout kept intact\nfinal eval + deployment demo", 0.58, 0.14, "#FDE68A"),
    ]
    for text, x, y, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#334155"},
        )
    ax.annotate("", xy=(0.46, 0.52), xytext=(0.22, 0.68), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.46, 0.52), xytext=(0.22, 0.35), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.46, 0.14), xytext=(0.22, 0.12), arrowprops={"arrowstyle": "->"})
    ax.set_title("Global test holdout rule")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_pipeline(path: Path) -> None:
    steps = [
        "P2 NPZ",
        "Load y + row_id",
        "Dirichlet train",
        "Dirichlet val",
        "Client artifacts",
        "Global test reference",
        "P3 reports",
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    x_positions = np.linspace(0.06, 0.94, len(steps))
    for idx, (x_pos, step) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EEF2FF", "edgecolor": "#4F46E5"},
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.045, 0.55),
                xytext=(x_pos + 0.045, 0.55),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.4},
            )
    ax.set_title("P3 Dirichlet pipeline L1/L2")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_markdown_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    accepted_text = "P3 est validée." if summary.get("accepted") else "P3 n'est pas validée."
    lines = [
        "# P3 — Dirichlet Split Report",
        "",
        "## 1. Objectif",
        "Créer des partitions fédérées non-IID pour L1 binaire et L2 family attack-only, sans entraîner de modèle.",
        "",
        "## 2. Entrées utilisées",
        f"- L1 train/val : `{summary['inputs']['l1_train_npz']}`, `{summary['inputs']['l1_val_npz']}`",
        f"- L2 train/val : `{summary['inputs']['l2_train_npz']}`, `{summary['inputs']['l2_val_npz']}`",
        "",
        "## 3. Rappel méthodologique Dirichlet",
        "Chaque classe est mélangée puis distribuée entre clients selon un tirage Dirichlet. Plus alpha est faible, plus les clients sont hétérogènes.",
        "",
        "## 4. Choix des alpha",
        "- `0.1` : non-IID extrême / stress-test.",
        "- `0.5` : scénario principal réaliste.",
        "- `5.0` : quasi-IID / référence stable.",
        "",
        "## 5. Choix des clients K",
        "Les scénarios couvrent `K ∈ {3, 4, 5}`.",
        "",
        "## 6. Règle deployment : global test holdout",
        "Train et val sont partitionnés pour FL. Le test global n’est pas partitionné et reste réservé à l’évaluation finale du modèle global, à la validation offline, aux tests microservices et à la simulation dashboard/inference.",
        "",
        "## 7. Partitionnement L1 binaire",
        f"- Scénarios générés : `{summary['l1']['scenario_count']}`.",
        "- Stockage : NPZ matérialisés par client pour train/val.",
        "",
        "## 8. Partitionnement L2 family attack-only",
        f"- Scénarios générés : `{summary['l2']['scenario_count']}`.",
        "- Stockage : index_only via `train_row_ids.npy` et `val_row_ids.npy`.",
        "",
        "## 9. Anti-leakage",
        f"- Scénarios valides : `{summary['anti_leakage']['valid_scenarios']}/{summary['anti_leakage']['total_scenarios']}`.",
        "",
        "## 10. Distributions par client",
        "Chaque scénario contient `distribution_report.json` et `client_distribution.csv`.",
        "",
        "## 11. Figures générées",
    ]
    for figure in summary["figures"]:
        lines.append(f"- `{figure}`")
    lines.extend(["", "## 12. Artefacts générés"])
    for artifact in summary["generated_artifacts"][:60]:
        lines.append(f"- `{artifact}`")
    if len(summary["generated_artifacts"]) > 60:
        lines.append(f"- ... `{len(summary['generated_artifacts']) - 60}` artefacts supplémentaires.")
    lines.extend(["", "## 13. Risques restants"])
    if summary["warnings"]:
        lines.extend([f"- {warning}" for warning in summary["warnings"]])
    else:
        lines.append("- Aucun warning restant.")
    lines.extend(["", "## 14. Critères d’acceptation", "", "| critere | ok |", "| --- | --- |"])
    for key, value in summary["criteria"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 15. Conclusion P3", "", accepted_text, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dirichlet_split(config_path: Path) -> DirichletRun:
    """Run P3 Dirichlet partition generation."""

    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    inputs = config["inputs"]
    dirichlet_cfg = config["dirichlet"]
    labels_cfg = config["labels"]
    outputs_cfg = config["outputs"]
    storage_cfg = config["storage"]

    final_dir = _repo_path(repo_root, config["final_experiment_dir"])
    partitions_dir = _repo_path(repo_root, outputs_cfg["partitions_dir"])
    reports_dir = _repo_path(repo_root, outputs_cfg["reports_dir"])
    figures_dir = _repo_path(repo_root, outputs_cfg["figures_dir"])
    partitions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    p2_check, errors, warnings = _verify_p2(repo_root, config)
    if errors:
        summary = {"accepted": False, "errors": errors, "warnings": warnings, "criteria": {}}
        profile = {"p2_check": p2_check, "errors": errors, "warnings": warnings}
        _write_json(reports_dir / "dirichlet_split_summary.json", summary)
        _write_json(reports_dir / "dirichlet_split_profile.json", profile)
        return DirichletRun(summary, profile, errors, warnings, [])

    alphas = [float(value) for value in dirichlet_cfg["alphas"]]
    client_counts = [int(value) for value in dirichlet_cfg["clients"]]
    seed = int(dirichlet_cfg["seed"])
    min_train_samples = int(dirichlet_cfg["min_train_samples_per_client"])
    max_attempts = int(dirichlet_cfg["max_resampling_attempts"])

    l1_class_names = _label_name_map(labels_cfg["l1_class_names"])
    l2_class_names = _label_name_map(labels_cfg["l2_family_names"])
    l1_target = str(labels_cfg["l1_target_key"])
    l2_target = str(labels_cfg["l2_target_key"])

    l1_train_path = _repo_path(repo_root, inputs["l1"]["train_npz"])
    l1_val_path = _repo_path(repo_root, inputs["l1"]["val_npz"])
    l1_test_path = _repo_path(repo_root, inputs["l1"]["test_npz"])
    l2_train_path = _repo_path(repo_root, inputs["l2"]["train_npz"])
    l2_val_path = _repo_path(repo_root, inputs["l2"]["val_npz"])
    l2_test_path = _repo_path(repo_root, inputs["l2"]["test_npz"])

    l1_train = _load_l1_npz(l1_train_path, l1_target)
    l1_val = _load_l1_npz(l1_val_path, l1_target)
    l2_train = _load_index_npz(l2_train_path, l2_target)
    l2_val = _load_index_npz(l2_val_path, l2_target)

    scenarios: list[dict[str, Any]] = []
    generated_files: list[str] = []
    scenario_warnings: list[str] = []

    for alpha in alphas:
        for num_clients in client_counts:
            scenario_seed = seed + int(alpha * 1000) + num_clients * 100

            l1_train_partition = dirichlet_partition_indices(
                l1_train[l1_target],
                num_clients,
                alpha,
                scenario_seed,
                min_train_samples,
                max_attempts,
            )
            l1_val_partition = dirichlet_partition_indices(
                l1_val[l1_target],
                num_clients,
                alpha,
                scenario_seed + 17,
                1,
                max_attempts,
            )
            l1_dir = _scenario_dir(partitions_dir, "l1_binary", alpha, num_clients)
            l1_result = _write_l1_scenario(
                repo_root=repo_root,
                scenario_dir=l1_dir,
                train_data=l1_train,
                val_data=l1_val,
                train_partition=l1_train_partition,
                val_partition=l1_val_partition,
                alpha=alpha,
                num_clients=num_clients,
                seed=seed,
                source_train_npz=l1_train_path,
                source_val_npz=l1_val_path,
                source_test_npz=l1_test_path,
                class_names=l1_class_names,
            )
            scenarios.append(
                {
                    "dataset_level": "l1_binary",
                    "alpha": alpha,
                    "num_clients": num_clients,
                    **l1_result,
                }
            )
            generated_files.extend([_rel(path, repo_root) for path in l1_result["files"]])
            scenario_warnings.extend(l1_result["manifest"]["warnings"])

            l2_train_partition = dirichlet_partition_indices(
                l2_train[l2_target],
                num_clients,
                alpha,
                scenario_seed + 31,
                min_train_samples,
                max_attempts,
            )
            l2_val_partition = dirichlet_partition_indices(
                l2_val[l2_target],
                num_clients,
                alpha,
                scenario_seed + 47,
                1,
                max_attempts,
            )
            l2_dir = _scenario_dir(partitions_dir, "l2_family", alpha, num_clients)
            l2_result = _write_l2_scenario(
                repo_root=repo_root,
                scenario_dir=l2_dir,
                train_data=l2_train,
                val_data=l2_val,
                train_partition=l2_train_partition,
                val_partition=l2_val_partition,
                alpha=alpha,
                num_clients=num_clients,
                seed=seed,
                source_train_npz=l2_train_path,
                source_val_npz=l2_val_path,
                source_test_npz=l2_test_path,
                class_names=l2_class_names,
            )
            scenarios.append(
                {
                    "dataset_level": "l2_family",
                    "alpha": alpha,
                    "num_clients": num_clients,
                    **l2_result,
                }
            )
            generated_files.extend([_rel(path, repo_root) for path in l2_result["files"]])
            scenario_warnings.extend(l2_result["manifest"]["warnings"])

    figures = {
        "l1_samples": figures_dir / "01_l1_samples_per_client_alpha_k.png",
        "l1_heatmaps": figures_dir / "02_l1_binary_heatmaps.png",
        "l1_alpha": figures_dir / "03_l1_alpha_comparison.png",
        "l2_samples": figures_dir / "04_l2_samples_per_client_alpha_k.png",
        "l2_heatmaps": figures_dir / "05_l2_family_heatmaps.png",
        "l2_alpha": figures_dir / "06_l2_alpha_comparison.png",
        "holdout": figures_dir / "07_global_test_holdout_explanation.png",
        "pipeline": figures_dir / "08_dirichlet_pipeline_l1_l2.png",
    }
    _figure_samples_per_client(scenarios, "l1_binary", figures["l1_samples"])
    _figure_heatmaps(scenarios, "l1_binary", l1_class_names, figures["l1_heatmaps"])
    _figure_alpha_comparison(scenarios, "l1_binary", figures["l1_alpha"])
    _figure_samples_per_client(scenarios, "l2_family", figures["l2_samples"])
    _figure_heatmaps(scenarios, "l2_family", l2_class_names, figures["l2_heatmaps"])
    _figure_alpha_comparison(scenarios, "l2_family", figures["l2_alpha"])
    _figure_global_test_holdout(figures["holdout"])
    _figure_pipeline(figures["pipeline"])
    generated_files.extend([_rel(path, repo_root) for path in figures.values()])

    valid_anti_leakage = sum(
        1 for scenario in scenarios if scenario["manifest"]["anti_leakage_result"]["valid"]
    )
    l1_scenarios = [scenario for scenario in scenarios if scenario["dataset_level"] == "l1_binary"]
    l2_scenarios = [scenario for scenario in scenarios if scenario["dataset_level"] == "l2_family"]
    generated_artifacts = [
        scenario["manifest"]["scenario_dir"] for scenario in scenarios
    ]
    warnings.extend(sorted(set(scenario_warnings)))

    criteria = {
        "p2_validated_detected": p2_check["p2_accepted"],
        "l1_9_scenarios_generated": len(l1_scenarios) == 9,
        "l2_9_scenarios_generated": len(l2_scenarios) == 9,
        "alphas_processed": sorted(alphas) == [0.1, 0.5, 5.0],
        "client_counts_processed": sorted(client_counts) == [3, 4, 5],
        "l1_train_val_partitioned": all(
            scenario["manifest"]["partition_train"] and scenario["manifest"]["partition_val"]
            for scenario in l1_scenarios
        ),
        "l1_test_global_not_partitioned": all(
            not scenario["manifest"]["partition_test"] for scenario in l1_scenarios
        ),
        "l1_global_test_references": all(
            Path(repo_root / scenario["manifest"]["global_test_reference_file"]).exists()
            for scenario in l1_scenarios
        ),
        "l1_client_npz_generated": all(
            all(
                (repo_root / files["train_scaled_npz"]).exists()
                and (repo_root / files["val_scaled_npz"]).exists()
                for files in scenario["manifest"]["client_files"].values()
            )
            for scenario in l1_scenarios
        ),
        "l2_train_val_index_only": all(
            scenario["manifest"]["storage_mode"] == "index_only"
            for scenario in l2_scenarios
        ),
        "l2_test_global_not_partitioned": all(
            not scenario["manifest"]["partition_test"] for scenario in l2_scenarios
        ),
        "l2_global_test_references": all(
            Path(repo_root / scenario["manifest"]["global_test_reference_file"]).exists()
            for scenario in l2_scenarios
        ),
        "l2_row_id_indexes_generated": all(
            all(
                (repo_root / files["train_row_ids_npy"]).exists()
                and (repo_root / files["val_row_ids_npy"]).exists()
                for files in scenario["manifest"]["client_files"].values()
            )
            for scenario in l2_scenarios
        ),
        "no_client_empty": all(
            all(
                counts["train"] > 0 and counts["val"] > 0
                for counts in scenario["manifest"]["row_counts"]["by_client"].values()
            )
            for scenario in scenarios
        ),
        "anti_leakage_valid": valid_anti_leakage == len(scenarios),
        "manifests_generated": all(
            (repo_root / scenario["manifest"]["output_files"]["manifest"]).exists()
            for scenario in scenarios
        ),
        "distribution_reports_generated": all(
            (repo_root / scenario["manifest"]["output_files"]["distribution_report"]).exists()
            for scenario in scenarios
        ),
        "client_distribution_csv_generated": all(
            (repo_root / scenario["manifest"]["output_files"]["client_distribution_csv"]).exists()
            for scenario in scenarios
        ),
        "figures_p3_generated": all(path.exists() for path in figures.values()),
    }
    accepted = all(criteria.values()) and not errors

    summary = {
        "accepted": accepted,
        "created_at": _created_at(),
        "inputs": {
            "l1_train_npz": _rel(l1_train_path, repo_root),
            "l1_val_npz": _rel(l1_val_path, repo_root),
            "l1_test_npz": _rel(l1_test_path, repo_root),
            "l2_train_npz": _rel(l2_train_path, repo_root),
            "l2_val_npz": _rel(l2_val_path, repo_root),
            "l2_test_npz": _rel(l2_test_path, repo_root),
        },
        "alphas": alphas,
        "clients": client_counts,
        "l1": {
            "scenario_count": len(l1_scenarios),
            "storage_mode": storage_cfg["l1_mode"],
            "global_test_holdout_path": _rel(l1_test_path, repo_root),
        },
        "l2": {
            "scenario_count": len(l2_scenarios),
            "storage_mode": storage_cfg["l2_mode"],
            "global_test_holdout_path": _rel(l2_test_path, repo_root),
        },
        "global_test_holdout": {
            "partition_test": False,
            "keep_global_test_holdout": True,
            "l1_test_path": _rel(l1_test_path, repo_root),
            "l2_test_path": _rel(l2_test_path, repo_root),
        },
        "anti_leakage": {
            "valid_scenarios": valid_anti_leakage,
            "total_scenarios": len(scenarios),
        },
        "scenario_summary": [
            {
                "dataset_level": scenario["dataset_level"],
                "alpha": scenario["alpha"],
                "num_clients": scenario["num_clients"],
                "scenario_dir": scenario["manifest"]["scenario_dir"],
                "train_total": scenario["manifest"]["row_counts"]["train_total"],
                "val_total": scenario["manifest"]["row_counts"]["val_total"],
                "anti_leakage_valid": scenario["manifest"]["anti_leakage_result"]["valid"],
                "warnings": scenario["manifest"]["warnings"],
            }
            for scenario in scenarios
        ],
        "figures": [_rel(path, repo_root) for path in figures.values()],
        "generated_artifacts": generated_artifacts,
        "criteria": criteria,
        "warnings": warnings,
        "errors": errors,
    }
    profile = {
        "p2_check": p2_check,
        "scenarios": [
            {
                "dataset_level": scenario["dataset_level"],
                "alpha": scenario["alpha"],
                "num_clients": scenario["num_clients"],
                "manifest": scenario["manifest"],
                "distribution_report": scenario["distribution_report"],
            }
            for scenario in scenarios
        ],
        "warnings": warnings,
        "errors": errors,
    }

    summary_path = reports_dir / "dirichlet_split_summary.json"
    profile_path = reports_dir / "dirichlet_split_profile.json"
    _write_json(summary_path, summary)
    _write_json(profile_path, profile)
    generated_files.extend([_rel(summary_path, repo_root), _rel(profile_path, repo_root)])

    docs_path = final_dir / "docs" / "03_dirichlet_split.md"
    _write_markdown_report(docs_path, summary)
    generated_files.append(_rel(docs_path, repo_root))

    return DirichletRun(
        summary=summary,
        profile=profile,
        errors=errors,
        warnings=warnings,
        generated_files=sorted(set(generated_files)),
    )
