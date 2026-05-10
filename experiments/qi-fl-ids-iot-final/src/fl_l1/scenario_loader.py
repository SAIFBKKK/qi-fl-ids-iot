"""Scenario loading helpers for P5 FedAvg L1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClientPartition:
    """Paths for one federated L1 client."""

    client_id: str
    train_npz: Path
    val_npz: Path
    train_samples: int
    val_samples: int


@dataclass(frozen=True)
class L1Scenario:
    """Resolved P3 L1 Dirichlet scenario."""

    alpha: float
    num_clients: int
    scenario_dir: Path
    manifest_path: Path
    distribution_report_path: Path
    global_test_reference_path: Path
    global_test_npz: Path
    clients: list[ClientPartition]
    manifest: dict[str, Any]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load P5 YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def repo_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repo-relative path."""

    return (repo_root / relative_path).resolve()


def rel(path: Path, repo_root: Path) -> str:
    """Return a stable repo-relative path."""

    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def alpha_dir(alpha: float) -> str:
    """Return P3 alpha directory name."""

    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def load_json(path: Path) -> Any:
    """Read JSON with UTF-8 encoding."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def scenario_dir_from_config(
    config: dict[str, Any],
    repo_root: Path,
    *,
    alpha: float,
    num_clients: int,
) -> Path:
    partitions_root = repo_path(repo_root, config["inputs"]["partitions_root"])
    return partitions_root / alpha_dir(alpha) / f"k{num_clients}"


def load_l1_scenario(
    config: dict[str, Any],
    repo_root: Path,
    *,
    alpha: float | None = None,
    num_clients: int | None = None,
) -> L1Scenario:
    """Load and validate one P3 L1 scenario without loading test data."""

    scenario_cfg = config["scenario"]
    resolved_alpha = float(alpha if alpha is not None else scenario_cfg["default_alpha"])
    resolved_clients = int(num_clients if num_clients is not None else scenario_cfg["default_k"])
    scenario_dir = scenario_dir_from_config(
        config,
        repo_root,
        alpha=resolved_alpha,
        num_clients=resolved_clients,
    )
    manifest_path = scenario_dir / "manifest.json"
    distribution_report_path = scenario_dir / "distribution_report.json"
    global_test_reference_path = scenario_dir / "global_test_reference.json"

    missing = [
        path
        for path in [scenario_dir, manifest_path, distribution_report_path, global_test_reference_path]
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"P3 L1 scenario is incomplete: {missing_text}")

    manifest = load_json(manifest_path)
    if manifest.get("dataset_level") != "l1_binary":
        raise ValueError(f"expected l1_binary manifest, got {manifest.get('dataset_level')!r}")
    if bool(manifest.get("partition_test", True)):
        raise ValueError("P5 refuses scenarios where test was partitioned")
    if not bool(manifest.get("keep_global_test_holdout", False)):
        raise ValueError("global test holdout flag is missing or false")

    reference = load_json(global_test_reference_path)
    global_test_npz = repo_path(repo_root, reference["global_test_npz"])
    configured_test = repo_path(repo_root, config["inputs"]["global_test_npz"])
    if global_test_npz.resolve() != configured_test.resolve():
        raise ValueError("scenario global_test_reference does not match P5 configured holdout")
    if not global_test_npz.exists():
        raise FileNotFoundError(f"global test holdout not found: {global_test_npz}")

    clients: list[ClientPartition] = []
    by_client = manifest["row_counts"]["by_client"]
    for client_index in range(1, resolved_clients + 1):
        client_id = f"client_{client_index}"
        client_dir = scenario_dir / client_id
        train_npz = client_dir / "train_scaled.npz"
        val_npz = client_dir / "val_scaled.npz"
        if not train_npz.exists() or not val_npz.exists():
            raise FileNotFoundError(f"missing train/val NPZ for {client_id}: {client_dir}")
        if (client_dir / "test_scaled.npz").exists():
            raise ValueError(f"client test partition found and forbidden: {client_dir}")
        clients.append(
            ClientPartition(
                client_id=client_id,
                train_npz=train_npz,
                val_npz=val_npz,
                train_samples=int(by_client[client_id]["train"]),
                val_samples=int(by_client[client_id]["val"]),
            )
        )

    return L1Scenario(
        alpha=resolved_alpha,
        num_clients=resolved_clients,
        scenario_dir=scenario_dir,
        manifest_path=manifest_path,
        distribution_report_path=distribution_report_path,
        global_test_reference_path=global_test_reference_path,
        global_test_npz=global_test_npz,
        clients=clients,
        manifest=manifest,
    )


def list_expected_scenarios(config: dict[str, Any]) -> list[tuple[float, int]]:
    """Return the configured alpha/K grid."""

    return [
        (float(alpha), int(num_clients))
        for alpha in config["scenario"]["alphas"]
        for num_clients in config["scenario"]["clients"]
    ]
