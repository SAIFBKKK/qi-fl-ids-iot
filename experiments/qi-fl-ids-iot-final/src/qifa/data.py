"""Scenario, mask, and run-path handling for QIFA."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fl_l1.client_data import ClientArrays, load_client_npz
from fl_l1.scenario_loader import L1Scenario, alpha_dir as p3_alpha_dir, load_l1_scenario
from qga.feature_mask import apply_feature_mask
from qifa.config import alpha_dir, load_json, rel, repo_path


@dataclass(frozen=True)
class QIFARunPaths:
    run_id: str
    scenario_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    latest_run_path: Path


def make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def configured_address(config: dict[str, Any], override: str | None = None) -> str:
    return str(override or config.get("flower", {}).get("address", "127.0.0.1:8085"))


def parse_address(address: str) -> tuple[str, int]:
    host, port = address.rsplit(":", 1)
    return host or "127.0.0.1", int(port)


def assert_port_available(address: str) -> None:
    host, port = parse_address(address)
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Flower server port is already in use: {host}:{port}. "
                f"Inspect with: netstat -ano | findstr :{port}"
            ) from exc


def load_scenario(config: dict[str, Any], repo_root: Path, *, alpha: float, clients: int) -> L1Scenario:
    bridge = {
        "inputs": {"partitions_root": config["inputs"]["partitions_root"], "global_test_npz": config["inputs"]["global_test_npz"]},
        "scenario": {"default_alpha": alpha, "default_k": clients, "alphas": [alpha], "clients": [clients]},
    }
    return load_l1_scenario(bridge, repo_root, alpha=alpha, num_clients=clients)


def load_mask_info(config: dict[str, Any], *, use_qga_mask: bool) -> dict[str, Any]:
    if not use_qga_mask:
        mask = np.ones(28, dtype=np.int8)
        return {
            "mask": mask,
            "selected_features": [],
            "selected_features_count": 28,
            "selected_mask_id": None,
            "selected_mask_source": "all_features",
            "calibration_decision_used": False,
        }
    final_dir = repo_path(config, "inputs.qga_final_mask_dir")
    feature_mask_path = final_dir / "feature_mask.json"
    selected_features_path = final_dir / "selected_features.json"
    decision_path = final_dir / "selection_decision.json"
    for path in [feature_mask_path, selected_features_path, decision_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing calibrated QGA mask artifact: {path}")
    feature_mask = load_json(feature_mask_path)
    selected = load_json(selected_features_path)
    decision = load_json(decision_path)
    mask = np.asarray(feature_mask["mask"], dtype=np.int8)
    selected_mask_id = str(decision.get("selected_mask_id"))
    if selected_mask_id != "conservative_seed_42":
        raise ValueError(f"Expected calibrated P8 mask conservative_seed_42, got {selected_mask_id!r}")
    features_count = int(mask.sum())
    if features_count != 12:
        raise ValueError(
            f"Expected QGA final mask with 12 selected features (conservative_seed_42), got {features_count}. "
            "Server and clients must use the same input_dim=12."
        )
    if features_count != int(decision.get("features_count", features_count)):
        raise ValueError("QGA final_selected_mask feature count mismatch")
    return {
        "mask": mask,
        "selected_features": selected.get("selected_features", feature_mask.get("selected_features", [])),
        "selected_features_count": int(mask.sum()),
        "selected_mask_id": selected_mask_id,
        "selected_mask_source": "final_selected_mask",
        "calibration_decision_used": True,
        "feature_mask_path": feature_mask_path,
        "selection_decision_path": decision_path,
    }


def _mask_arrays(arrays: ClientArrays, mask: np.ndarray) -> ClientArrays:
    return ClientArrays(
        X=apply_feature_mask(arrays.X, mask).astype(np.float32, copy=False),
        y=arrays.y,
        label_id_original=arrays.label_id_original,
        row_id=arrays.row_id,
    )


def load_client_arrays(scenario: L1Scenario, *, client_id: str, mask: np.ndarray, max_samples: int | None, seed: int) -> tuple[ClientArrays, ClientArrays]:
    partition = next(client for client in scenario.clients if client.client_id == client_id)
    train = load_client_npz(partition.train_npz, max_samples=max_samples, seed=seed)
    val = load_client_npz(partition.val_npz, max_samples=max_samples, seed=seed + 10_000)
    return _mask_arrays(train, mask), _mask_arrays(val, mask)


def concatenate_validation_arrays(scenario: L1Scenario, *, mask: np.ndarray, max_samples_per_client: int | None, seed: int) -> ClientArrays:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    for index, partition in enumerate(scenario.clients, start=1):
        arrays = load_client_npz(partition.val_npz, max_samples=max_samples_per_client, seed=seed + 20_000 + index)
        masked = _mask_arrays(arrays, mask)
        xs.append(masked.X)
        ys.append(masked.y)
        label_ids.append(masked.label_id_original)
        row_ids.append(masked.row_id)
    return ClientArrays(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        label_id_original=np.concatenate(label_ids, axis=0),
        row_id=np.concatenate(row_ids, axis=0),
    )


def prepare_run_paths(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    variant: str,
    gamma: float,
    use_qga_mask: bool = False,
    run_id: str | None = None,
    mark_latest: bool = True,
) -> QIFARunPaths:
    feature_mode = "qga_mask" if use_qga_mask else "full_features"
    resolved_run_id = run_id or make_run_id()
    scenario_dir = repo_path(config, "outputs.run_dir") / feature_mode / alpha_dir(alpha) / f"k{clients}" / f"variant_{variant}" / f"gamma_{gamma}"
    run_dir = scenario_dir / "runs" / resolved_run_id
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    for path in [checkpoints_dir, artifacts_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    latest_run_path = scenario_dir / "latest_run.json"
    if mark_latest:
        latest_run_path.write_text(
            json.dumps(
                {
                    "run_id": resolved_run_id,
                    "run_dir": rel(run_dir, repo_root),
                    "latest_run_summary": rel(scenario_dir / "latest_run_summary.json", repo_root),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return QIFARunPaths(
        run_id=resolved_run_id,
        scenario_dir=scenario_dir,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        latest_run_path=latest_run_path,
    )


def latest_run_id(*, config: dict[str, Any], repo_root: Path, alpha: float, clients: int, variant: str, gamma: float, use_qga_mask: bool = False) -> str:
    feature_mode = "qga_mask" if use_qga_mask else "full_features"
    path = repo_path(config, "outputs.run_dir") / feature_mode / alpha_dir(alpha) / f"k{clients}" / f"variant_{variant}" / f"gamma_{gamma}" / "latest_run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["run_id"])
