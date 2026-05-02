from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import flwr
import numpy as np
import pandas as pd
import torch
import yaml
from flwr.simulation import run_simulation
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATASET_CSV, DATASET_PARQUET, OUTPUTS_DIR
from src.common.utils import get_expected_node_ids, set_seed
from src.fl.client_app import create_client_app
from src.fl.server_app import create_server_app
from src.model.network import MLPClassifier
from src.scripts.generate_scenarios import run_scenario
from src.scripts.generate_weights import compute_weights, load_global_counts, output_path_for_scenario
from src.tracking.artifact_logger import BaselineArtifactTracker


LOGGER = get_logger("model_factory")
NODE_IDS = ("node1", "node2", "node3")
REQUIRED_SPLITS = ("train", "val", "test")
LABEL_MAPPING_PATH = ARTIFACTS_DIR / "baseline" / "artifacts" / "label_mapping_34.pkl"
REPORT_PATH = ROOT.parents[1] / "docs" / "reports" / "MODEL_FACTORY_30ROUNDS_REPORT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return cfg


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def print_environment() -> dict[str, Any]:
    payload = {
        "sys_executable": sys.executable,
        "flwr_version": flwr.__version__,
        "flwr_file": str(Path(flwr.__file__).resolve()),
        "venv_warning": ".venv" not in str(Path(sys.executable)).lower(),
    }
    print("[model_factory] Python executable:", payload["sys_executable"])
    print("[model_factory] Flower version:", payload["flwr_version"])
    print("[model_factory] Flower file:", payload["flwr_file"])
    if payload["venv_warning"]:
        print("[model_factory][WARNING] Python executable is not inside .venv")
    return payload


def write_run_status(
    output_root: Path,
    *,
    status: str,
    started_at: str,
    models: list[str],
    rounds: int,
    smoke_test: bool,
    finished_at: str | None = None,
    error: str | None = None,
    environment: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "models": models,
        "rounds": rounds,
        "smoke_test": smoke_test,
    }
    if error:
        payload["error"] = error
    if environment is not None:
        payload["environment"] = environment
    dump_json(output_root / "run_status.json", payload)


def manifest_path(scenario: str) -> Path:
    return ROOT / "data" / "splits" / f"{scenario}_manifest.json"


def raw_path(scenario: str, node_id: str, split: str) -> Path:
    return ROOT / "data" / "raw" / scenario / node_id / f"{split}.csv"


def processed_path(scenario: str, node_id: str, split: str) -> Path:
    return ROOT / "data" / "processed" / scenario / node_id / f"{split}_preprocessed.npz"


def scaler_path(scenario: str) -> Path:
    return ARTIFACTS_DIR / f"scaler_standard_train_{scenario}.pkl"


def feature_names_path(scenario: str) -> Path:
    return ARTIFACTS_DIR / f"feature_names_{scenario}.pkl"


def parse_models_arg(raw: str | None, available: dict[str, Any], smoke_test: bool) -> list[str]:
    if raw is None:
        if smoke_test:
            raise ValueError("--models is required when --smoke-test is enabled.")
        return [name for name in ("weak", "medium", "powerful") if name in available]

    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if not selected:
        raise ValueError("--models must contain at least one model name.")

    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(available)}")
    return selected


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "smoke"


def compute_counts_from_smoke_npz(scenario: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node_id in NODE_IDS:
        data = np.load(processed_path(scenario, node_id, "train"), allow_pickle=True)
        values, freqs = np.unique(data["y"], return_counts=True)
        for value, freq in zip(values, freqs):
            cls = int(value)
            counts[cls] = counts.get(cls, 0) + int(freq)
    return counts


def write_smoke_manifest(
    *,
    base_scenario: str,
    smoke_scenario: str,
    max_samples_per_client: int,
    feature_names: list[str],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "scenario": smoke_scenario,
        "source_scenario": base_scenario,
        "smoke_test": True,
        "max_samples_per_client": int(max_samples_per_client),
        "label_column": "label_id",
        "row_id_column": "__row_id",
        "num_nodes": len(NODE_IDS),
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "preprocessing": {
            "scaler": "StandardScaler",
            "fit_split": "train",
            "feature_count": len(feature_names),
            "feature_names_artifact": str(feature_names_path(smoke_scenario)),
            "scaler_artifact": str(scaler_path(smoke_scenario)),
        },
        "splits": {},
    }

    for split in REQUIRED_SPLITS:
        manifest["splits"][split] = {"type": "smoke_subset", "nodes": {}}
        for node_id in NODE_IDS:
            raw_csv = raw_path(smoke_scenario, node_id, split)
            npz_path = processed_path(smoke_scenario, node_id, split)
            data = np.load(npz_path, allow_pickle=True)
            y = np.asarray(data["y"], dtype=np.int64)
            values, freqs = np.unique(y, return_counts=True)
            manifest["splits"][split]["nodes"][node_id] = {
                "raw_csv": str(raw_csv),
                "processed_npz": str(npz_path),
                "rows": int(len(y)),
                "class_distribution": {str(int(v)): int(f) for v, f in zip(values, freqs)},
            }

    manifest_path(smoke_scenario).parent.mkdir(parents=True, exist_ok=True)
    dump_json(manifest_path(smoke_scenario), manifest)
    return manifest


def prepare_smoke_scenario(
    *,
    base_scenario: str,
    output_root: Path,
    max_samples_per_client: int,
) -> tuple[str, dict[str, Any]]:
    if max_samples_per_client < 1:
        raise ValueError("--max-samples-per-client must be >= 1.")

    smoke_scenario = f"{base_scenario}_smoke_{safe_name(output_root.name)}"
    with feature_names_path(base_scenario).open("rb") as handle:
        feature_names = list(pickle.load(handle))
    with scaler_path(base_scenario).open("rb") as handle:
        scaler = pickle.load(handle)

    LOGGER.info(
        "Preparing smoke scenario %s from %s with max %d samples/client/split",
        smoke_scenario,
        base_scenario,
        max_samples_per_client,
    )
    for split in REQUIRED_SPLITS:
        for node_id in NODE_IDS:
            source_csv = raw_path(base_scenario, node_id, split)
            if not source_csv.exists():
                raise FileNotFoundError(source_csv)

            raw_subset = pd.read_csv(source_csv, nrows=max_samples_per_client)
            y = raw_subset["label_id"].to_numpy(dtype=np.int64)
            X_raw = raw_subset[feature_names].to_numpy(dtype=np.float64)
            X_scaled = scaler.transform(X_raw).astype(np.float32)

            dest_raw = raw_path(smoke_scenario, node_id, split)
            dest_npz = processed_path(smoke_scenario, node_id, split)
            dest_raw.parent.mkdir(parents=True, exist_ok=True)
            dest_npz.parent.mkdir(parents=True, exist_ok=True)
            raw_subset.to_csv(dest_raw, index=False)
            np.savez_compressed(
                dest_npz,
                X=X_scaled,
                y=y,
                feature_names=np.array(feature_names, dtype=object),
            )

    shutil.copy2(scaler_path(base_scenario), scaler_path(smoke_scenario))
    shutil.copy2(feature_names_path(base_scenario), feature_names_path(smoke_scenario))

    counts = compute_counts_from_smoke_npz(smoke_scenario)
    weights = compute_weights(counts, num_classes=34)
    with output_path_for_scenario(smoke_scenario).open("wb") as handle:
        pickle.dump(weights, handle)

    manifest = write_smoke_manifest(
        base_scenario=base_scenario,
        smoke_scenario=smoke_scenario,
        max_samples_per_client=max_samples_per_client,
        feature_names=feature_names,
    )
    return smoke_scenario, manifest


def validate_manifest(scenario: str, split_cfg: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(scenario)
    if not path.exists():
        raise FileNotFoundError(path)

    manifest = load_json(path)
    ratios = manifest.get("split_ratios", {})
    expected = {
        "train": float(split_cfg.get("train", 0.70)),
        "val": float(split_cfg.get("validation", 0.15)),
        "test": float(split_cfg.get("deployment", 0.15)),
    }
    actual = {
        "train": float(ratios.get("train", -1.0)),
        "val": float(ratios.get("val", -1.0)),
        "test": float(ratios.get("test", -1.0)),
    }
    for key, expected_value in expected.items():
        if abs(actual[key] - expected_value) > 1e-9:
            raise ValueError(f"{path} split ratio {key}={actual[key]} != {expected_value}")

    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("fit_split") != "train":
        raise ValueError(f"{path} does not declare train-only scaler fitting.")
    if int(preprocessing.get("feature_count", 0)) != 28:
        raise ValueError(f"{path} feature_count must be 28.")
    if "splits" not in manifest:
        raise ValueError(f"{path} is not a split-aware v3 manifest.")

    missing: list[str] = []
    for split in REQUIRED_SPLITS:
        for node_id in NODE_IDS:
            for candidate in (raw_path(scenario, node_id, split), processed_path(scenario, node_id, split)):
                if not candidate.exists():
                    missing.append(str(candidate))
    if missing:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing[:20]))

    return manifest


def ensure_scenario_ready(scenario: str, split_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    try:
        manifest = validate_manifest(scenario, split_cfg)
        LOGGER.info("Reusing validated split manifest: %s", manifest_path(scenario))
        return manifest
    except FileNotFoundError as exc:
        if not DATASET_PARQUET.exists() and not DATASET_CSV.exists():
            raise FileNotFoundError(
                f"Scenario {scenario!r} is incomplete and source dataset is unavailable. "
                f"Tried {DATASET_PARQUET} and {DATASET_CSV}."
            ) from exc

        LOGGER.warning("Scenario %s is incomplete; regenerating from source dataset.", scenario)
        run_scenario(scenario=scenario, seed=seed)
        return validate_manifest(scenario, split_cfg)


def ensure_class_weights(scenario: str) -> Path:
    output_path = output_path_for_scenario(scenario)
    if output_path.exists():
        LOGGER.info("Reusing class weights: %s", output_path)
        return output_path

    counts = load_global_counts(scenario, split="train")
    weights = compute_weights(counts, num_classes=34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(weights, handle)
    LOGGER.info("Class weights generated: %s", output_path)
    return output_path


def build_fl_config(
    *,
    model_name: str,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    scenario: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment_prefix = str(training_cfg.get("experiment_prefix", "model_factory_30rounds"))
    experiment = {
        "name": f"{experiment_prefix}_{model_name}",
        "architecture": f"model_factory_{model_name}",
        "fl_strategy": str(training_cfg.get("strategy", "fedavg")),
        "data_scenario": scenario,
        "imbalance_strategy": "class_weights" if training_cfg.get("use_class_weights", True) else "none",
        "description": f"Model factory {model_name} FL run.",
    }
    hidden_layers = [int(v) for v in model_cfg["hidden_layers"]]
    config = {
        "project": {
            "name": "fl-iot-ids-v3",
            "seed": int(training_cfg.get("seed", 42)),
        },
        "dataset": {
            "feature_count": int(model_cfg.get("input_dim", 28)),
            "num_classes": int(model_cfg.get("output_dim", 34)),
            "benign_class_id": 1,
            "rare_class_ids": [0, 3, 30, 31, 33],
        },
        "scenario": {
            "name": scenario,
            "num_clients": 3,
            "node_ids": list(NODE_IDS),
        },
        "strategy": {
            "name": str(training_cfg.get("strategy", "fedavg")),
            "num_rounds": int(training_cfg.get("rounds", 30)),
            "fraction_train": 1.0,
            "fraction_evaluate": 1.0,
            "min_train_nodes": 3,
            "min_evaluate_nodes": 3,
            "min_available_nodes": 3,
            "expert_factor": 1.0,
            "expert_node_id": "node3",
            "multitier_enabled": False,
        },
        "train": {
            "local_epochs": int(training_cfg.get("local_epochs", 1)),
            "batch_size": int(training_cfg.get("batch_size", 256)),
            "learning_rate": float(training_cfg.get("learning_rate", 0.0005)),
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "proximal_mu": 0.0,
        },
        "model": {
            "name": f"model_factory_{model_name}",
            "input_dim": int(model_cfg.get("input_dim", 28)),
            "hidden_dims": hidden_layers,
            "output_dim": int(model_cfg.get("output_dim", 34)),
            "dropout": float(training_cfg.get("dropout", 0.2)),
        },
        "imbalance": {
            "name": "class_weights" if training_cfg.get("use_class_weights", True) else "none",
            "enabled": bool(training_cfg.get("use_class_weights", True)),
            "loss": "weighted_cross_entropy" if training_cfg.get("use_class_weights", True) else "cross_entropy",
        },
        "evaluation": {
            "centralized_eval": False,
            "federated_eval": True,
            "best_round_monitor": "macro_f1",
        },
        "mlflow": {
            "enabled": False,
            "tracking_uri": "./outputs/mlruns",
        },
    }
    return experiment, config


def run_fl_model(model_name: str, model_cfg: dict[str, Any], training_cfg: dict[str, Any], scenario: str) -> Path:
    experiment, config = build_fl_config(
        model_name=model_name,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        scenario=scenario,
    )
    set_seed(int(config["project"]["seed"]))
    config["scenario"]["node_ids"] = get_expected_node_ids(int(config["scenario"]["num_clients"]))

    tracker = BaselineArtifactTracker(experiment=experiment, config=config)
    report_dir = tracker.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report_dir / "resolved_config.json", {"experiment": experiment, "config": config})

    LOGGER.info("Starting FL simulation: %s", experiment["name"])
    server_app = create_server_app(config, tracker=tracker)
    client_app = create_client_app(config)
    backend_config: dict[str, Any] = {
        "init_args": {"include_dashboard": False},
        "client_resources": {"num_cpus": 1},
    }
    if sys.platform.startswith("win"):
        backend_config["init_args"]["num_cpus"] = 1

    started = perf_counter()
    status = "success"
    error_message: str | None = None
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=int(config["scenario"]["num_clients"]),
            backend_name="ray",
            backend_config=backend_config,
        )
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        tracker.save_baseline_artifacts(
            status=status,
            duration_sec=perf_counter() - started,
            error_message=error_message,
        )

    checkpoint = report_dir / "best_checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best checkpoint produced for {model_name}: {checkpoint}")

    summary_path = report_dir / "run_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        requested_rounds = int(summary.get("requested_rounds", config["strategy"]["num_rounds"]))
        if summary.get("status") == "partial" and int(summary.get("rounds", 0)) >= requested_rounds:
            summary["status"] = "success"
            summary["completed_rounds"] = requested_rounds
            summary["status_note"] = (
                "Normalized by run_model_factory after Flower completed and "
                "best_checkpoint.pth was produced."
            )
            dump_json(summary_path, summary)
    return report_dir


def load_label_mapping_json() -> dict[str, Any]:
    with LABEL_MAPPING_PATH.open("rb") as handle:
        mapping = pickle.load(handle)
    label_to_id = mapping.get("label_to_id", {})
    id_to_label = mapping.get("id_to_label", {})
    return {
        "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
        "id_to_label": {str(k): str(v) for k, v in id_to_label.items()},
    }


def load_npz_split(scenario: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for node_id in NODE_IDS:
        data = np.load(processed_path(scenario, node_id, split), allow_pickle=True)
        X_parts.append(np.asarray(data["X"], dtype=np.float32))
        y_parts.append(np.asarray(data["y"], dtype=np.int64))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def evaluate_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model = MLPClassifier(
        input_dim=int(model_cfg.get("input_dim", 28)),
        num_classes=int(model_cfg.get("output_dim", 34)),
        hidden_dims=[int(v) for v in model_cfg["hidden_layers"]],
        dropout=0.0,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32)
            logits = model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    benign_id = 1
    benign_mask = y == benign_id
    attack_mask = y != benign_id
    benign_recall = float(recall_score(y[benign_mask], y_pred[benign_mask], labels=[benign_id], average="micro", zero_division=0)) if benign_mask.any() else 0.0
    attack_recall = float((y_pred[attack_mask] != benign_id).mean()) if attack_mask.any() else 0.0
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "benign_recall": benign_recall,
        "attack_recall": attack_recall,
        "false_positive_rate": float(1.0 - benign_recall),
        "num_samples": float(len(y)),
    }


def export_deployment_data(output_root: Path, scenario: str, manifest: dict[str, Any]) -> dict[str, Any]:
    deployment_dir = output_root / "deployment_data"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = [pd.read_csv(raw_path(scenario, node_id, "test")) for node_id in NODE_IDS]
    deployment_df = pd.concat(raw_frames, ignore_index=True)
    deployment_path = deployment_dir / "deployment_15.parquet"
    deployment_df.to_parquet(deployment_path, index=False)

    shutil.copy2(feature_names_path(scenario), deployment_dir / "feature_names.pkl")
    dump_json(deployment_dir / "label_mapping.json", load_label_mapping_json())

    train_rows = sum(
        int(node["rows"]) for node in manifest["splits"]["train"]["nodes"].values()
    )
    val_rows = sum(int(node["rows"]) for node in manifest["splits"]["val"]["nodes"].values())
    test_rows = int(len(deployment_df))
    split_summary = {
        "generated_at": utc_now(),
        "scenario": scenario,
        "source": "data/raw/{scenario}/node*/test.csv",
        "ratios": {"train": 0.70, "validation": 0.15, "deployment": 0.15},
        "rows": {"train": train_rows, "validation": val_rows, "deployment": test_rows},
        "deployment_file": str(deployment_path),
        "anti_leakage": {
            "deployment_used_for_training": False,
            "deployment_used_for_validation": False,
            "fl_partitions_source": "train split only",
            "validation_source": "val split only",
            "deployment_source": "test split only",
            "row_id_column": "__row_id",
        },
    }
    dump_json(deployment_dir / "split_summary.json", split_summary)
    return split_summary


def export_bundle(
    *,
    output_root: Path,
    model_name: str,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    scenario: str,
    report_dir: Path,
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    bundle_dir = output_root / model_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(report_dir / "best_checkpoint.pth", map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_path = bundle_dir / "global_model.pth"
    torch.save(state_dict, model_path)
    shutil.copy2(scaler_path(scenario), bundle_dir / "scaler.pkl")
    shutil.copy2(feature_names_path(scenario), bundle_dir / "feature_names.pkl")
    dump_json(bundle_dir / "label_mapping.json", load_label_mapping_json())

    run_summary = load_json(report_dir / "run_summary.json")
    dump_json(bundle_dir / "run_summary.json", run_summary)
    metrics = {
        "generated_at": utc_now(),
        "model": model_name,
        "validation": validation_metrics,
        "federated_summary": {
            key: value
            for key, value in run_summary.items()
            if key.startswith("final_") or key in {"completed_rounds", "requested_rounds", "status"}
        },
    }
    dump_json(bundle_dir / "metrics.json", metrics)

    sha256s = {
        name: sha256_file(bundle_dir / name)
        for name in ("global_model.pth", "scaler.pkl", "feature_names.pkl")
    }
    model_config = {
        "model_name": model_name,
        "input_dim": int(model_cfg.get("input_dim", 28)),
        "hidden_layers": [int(v) for v in model_cfg["hidden_layers"]],
        "hidden_dims": [int(v) for v in model_cfg["hidden_layers"]],
        "output_dim": int(model_cfg.get("output_dim", 34)),
        "num_classes": int(model_cfg.get("output_dim", 34)),
        "dropout": float(training_cfg.get("dropout", 0.2)),
        "fl_strategy": str(training_cfg.get("strategy", "fedavg")),
        "rounds": int(training_cfg.get("rounds", 30)),
        "local_epochs": int(training_cfg.get("local_epochs", 1)),
        "use_class_weights": bool(training_cfg.get("use_class_weights", True)),
        "scenario": scenario,
        "checkpoint_round": int(checkpoint.get("round", 0)),
        "bundle_built_at": utc_now(),
        "sha256": sha256s,
    }
    dump_json(bundle_dir / "model_config.json", model_config)
    return metrics


def write_report(output_root: Path, config: dict[str, Any], results: dict[str, Any], split_summary: dict[str, Any]) -> None:
    rows = []
    for name, payload in results.items():
        metrics = payload.get("validation", {})
        model_path = output_root / name / "global_model.pth"
        rows.append(
            "| {name} | {acc:.6f} | {macro:.6f} | {weighted:.6f} | {benign:.6f} | {attack:.6f} | {fpr:.6f} | {size:.1f} KB |".format(
                name=name,
                acc=float(metrics.get("accuracy", 0.0)),
                macro=float(metrics.get("macro_f1", 0.0)),
                weighted=float(metrics.get("weighted_f1", 0.0)),
                benign=float(metrics.get("benign_recall", 0.0)),
                attack=float(metrics.get("attack_recall", 0.0)),
                fpr=float(metrics.get("false_positive_rate", 0.0)),
                size=model_path.stat().st_size / 1024 if model_path.exists() else 0.0,
            )
        )

    content = f"""# Model Factory 30 Rounds Report

Generated at: `{utc_now()}`

## 1. Resume executif

Trois modeles IDS offline ont ete entraines avec la pipeline `experiments/fl-iot-ids-v3` existante: `weak`, `medium`, `powerful`. Les runs utilisent Flower/FedAvg, 30 rounds, 3 clients, class weights, 28 features et 34 classes.

## 2. Fonctionnement actuel de fl-iot-ids-v3

La pipeline existante splitte les lignes raw avant preprocessing, fit le scaler sur `train`, cree des partitions FL par scenario, entraine avec `src.fl.client_app`/`src.fl.server_app`, evalue sur `val_preprocessed.npz`, puis sauvegarde checkpoints et summaries via `BaselineArtifactTracker`.

## 3. Scripts reutilises

`generate_scenarios.py`, `prepare_partitions.py`, `generate_weights.py`, `client_app.py`, `server_app.py`, `reporting_strategy.py`, `evaluate.py`, `losses.py`, `dataloader.py`, `artifact_logger.py`.

## 4. Changements effectues

Ajout de `configs/model_factory_30rounds.yaml`, `scripts/run_model_factory.py`, `scripts/validate_model_factory_bundles.py`; generalisation minimale de `MLPClassifier` aux listes de couches cachees.

## 5. Split 70/15/15

Scenario: `{config.get("training", {}).get("scenario", "normal_noniid")}`. Rows: train={split_summary["rows"]["train"]}, validation={split_summary["rows"]["validation"]}, deployment={split_summary["rows"]["deployment"]}.

## 6. Verification anti data leakage

Le scaler declare `fit_split=train`; les partitions FL proviennent uniquement de `train_preprocessed.npz`; l'evaluation Flower utilise `val_preprocessed.npz`; `deployment_15.parquet` est exporte depuis `test.csv` uniquement et n'est jamais charge par le training.

## 7. Architectures

- weak: 28 -> 64 -> 34
- medium: 28 -> 128 -> 64 -> 34
- powerful: 28 -> 256 -> 128 -> 34

## 8. Configuration FL

FedAvg, 30 rounds, 1 local epoch, class weights actives, 3 clients.

## 9-10. Metriques validation et comparaison

| model | accuracy | macro-F1 | weighted-F1 | benign recall | attack recall | FPR | taille |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Latence estimee: non mesuree dans ce run offline.

## 11. Emplacement des bundles

`{output_root}`

## 12. Utilisation future dans Mode A

Chaque sous-dossier contient `global_model.pth`, `scaler.pkl`, `feature_names.pkl`, `label_mapping.json`, `model_config.json`, `run_summary.json`, `metrics.json`. Mode A pourra charger le bundle choisi, reconstruire `MLPClassifier` avec `hidden_layers`, appliquer `scaler.pkl` dans l'ordre `feature_names.pkl`, puis mapper la prediction avec `label_mapping.json`.

## 13. Limitations et prochaines etapes

Le dataset source externe n'etait pas visible ici; le run s'appuie donc sur les splits `normal_noniid` deja prepares et valides. Prochaine etape: brancher explicitement le choix weak/medium/powerful dans Mode A sans modifier les garanties anti-leakage.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weak/medium/powerful FL IDS bundles.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list, e.g. weak or weak,medium.")
    parser.add_argument("--rounds", type=int, default=None, help="Override configured FL rounds.")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even when bundle files already exist.")
    args = parser.parse_args()

    environment = print_environment()
    config = load_config(args.config)
    models = dict(config.get("models", {}))
    training_cfg = dict(config.get("training", {}))
    split_cfg = dict(config.get("split", {}))
    selected_models = parse_models_arg(args.models, models, smoke_test=bool(args.smoke_test))
    if args.rounds is not None:
        if args.rounds < 1:
            raise ValueError("--rounds must be >= 1.")
        training_cfg["rounds"] = int(args.rounds)
    rounds = int(training_cfg.get("rounds", 30))

    if args.smoke_test and args.output is None:
        raise ValueError("--output is required when --smoke-test is enabled.")
    output_arg = args.output or (OUTPUTS_DIR / "model_factory_30rounds")
    output_root = output_arg if output_arg.is_absolute() else (Path.cwd() / output_arg)
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    write_run_status(
        output_root,
        status="running",
        started_at=started_at,
        models=selected_models,
        rounds=rounds,
        smoke_test=bool(args.smoke_test),
        environment=environment,
    )

    base_scenario = str(training_cfg.get("scenario", "normal_noniid"))
    seed = int(training_cfg.get("seed", 42))
    scenario = base_scenario
    if args.smoke_test:
        training_cfg["experiment_prefix"] = f"model_factory_smoke_{safe_name(output_root.name)}"
    config["training"] = training_cfg

    try:
        if args.smoke_test and args.max_samples_per_client is None:
            raise ValueError("--max-samples-per-client is required when --smoke-test is enabled.")

        manifest = ensure_scenario_ready(base_scenario, split_cfg, seed=seed)
        ensure_class_weights(base_scenario)

        if args.smoke_test:
            scenario, manifest = prepare_smoke_scenario(
                base_scenario=base_scenario,
                output_root=output_root,
                max_samples_per_client=int(args.max_samples_per_client),
            )
        ensure_class_weights(scenario)

        X_val, y_val = load_npz_split(scenario, "val")
        results: dict[str, Any] = {}

        for model_name in selected_models:
            model_cfg = dict(models[model_name])
            bundle_dir = output_root / model_name
            required = [
                bundle_dir / "global_model.pth",
                bundle_dir / "scaler.pkl",
                bundle_dir / "feature_names.pkl",
                bundle_dir / "label_mapping.json",
                bundle_dir / "model_config.json",
                bundle_dir / "run_summary.json",
                bundle_dir / "metrics.json",
            ]
            if not args.force and all(path.exists() for path in required):
                LOGGER.info("Skipping existing bundle: %s", bundle_dir)
                results[model_name] = load_json(bundle_dir / "metrics.json")
                continue

            report_dir = run_fl_model(model_name, model_cfg, training_cfg, scenario)
            checkpoint = torch.load(report_dir / "best_checkpoint.pth", map_location="cpu", weights_only=False)
            validation_metrics = evaluate_state_dict(
                checkpoint["state_dict"],
                model_cfg,
                X_val,
                y_val,
                batch_size=int(training_cfg.get("batch_size", 256)),
            )
            metrics = export_bundle(
                output_root=output_root,
                model_name=model_name,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                scenario=scenario,
                report_dir=report_dir,
                validation_metrics=validation_metrics,
            )
            results[model_name] = metrics

        split_summary = export_deployment_data(output_root, scenario, manifest)
        write_report(output_root, config, results, split_summary)
        dump_json(
            output_root / "model_factory_summary.json",
            {
                "generated_at": utc_now(),
                "config": str(args.config),
                "output": str(output_root),
                "models": results,
                "deployment": split_summary,
                "report": str(REPORT_PATH),
                "smoke_test": bool(args.smoke_test),
            },
        )
        write_run_status(
            output_root,
            status="completed",
            started_at=started_at,
            finished_at=utc_now(),
            models=selected_models,
            rounds=rounds,
            smoke_test=bool(args.smoke_test),
            environment=environment,
        )
        LOGGER.info("Model factory completed: %s", output_root)
    except Exception as exc:
        write_run_status(
            output_root,
            status="failed",
            started_at=started_at,
            finished_at=utc_now(),
            models=selected_models,
            rounds=rounds,
            smoke_test=bool(args.smoke_test),
            error=str(exc),
            environment=environment,
        )
        raise


if __name__ == "__main__":
    main()
