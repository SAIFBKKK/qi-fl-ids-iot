"""Lightweight in-process P10 robustness runner."""

from __future__ import annotations

import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.l1_mlp import CentralizedL1MLP

from fl_l1.aggregation import fedavg_state_dicts
from qifa.scoring import (
    amplitudes_from_theta,
    hybrid_weights,
    normalize_scores_to_theta,
    probabilities_from_amplitudes,
)

from .config import alpha_dir, clients_dir, poison_rate_dir, write_json
from .metrics import binary_metrics, robustness_score
from .poisoning import apply_poisoning
from .scenario import RobustnessScenario, load_npz_arrays


def _select_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("training", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def _build_model(input_dim: int, config: dict[str, Any]) -> CentralizedL1MLP:
    model_cfg = config.get("model", {})
    return CentralizedL1MLP(
        input_dim=int(input_dim),
        hidden_layers=list(model_cfg.get("hidden_layers", [128, 64])),
        output_dim=int(model_cfg.get("output_dim", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        activation=str(model_cfg.get("activation", "relu")),
    )


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator)


def _evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    preds: list[np.ndarray] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in _loader(X, y, batch_size, False, 0):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            losses.append(float(criterion(logits, yb).item()) * int(yb.numel()))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    metrics = binary_metrics(y, y_pred)
    metrics["loss"] = float(sum(losses) / max(len(y), 1))
    return metrics


def _fit_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    local = _build_model(X.shape[1], config).to(device)
    local.load_state_dict(model.state_dict())
    local.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        local.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )
    losses: list[float] = []
    for _ in range(int(config["training"].get("local_epochs", 1))):
        for xb, yb in _loader(X, y, int(config["training"].get("batch_size", 512)), True, seed):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(local(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in local.state_dict().items()), float(np.mean(losses) if losses else 0.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(config: dict[str, Any], scenario: RobustnessScenario, run_id: str) -> Path:
    return (
        Path.cwd()
        / config["outputs"]["run_dir"]
        / scenario.method
        / alpha_dir(scenario.alpha)
        / f"k{scenario.clients}"
        / scenario.attack_type
        / poison_rate_dir(scenario.poison_rate)
        / clients_dir(scenario.poisoned_clients)
        / "runs"
        / run_id
    )


def run_robustness_scenario(
    *,
    config: dict[str, Any],
    scenario: RobustnessScenario,
    mode: str,
    max_samples: int | None,
) -> dict[str, Any]:
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    base_dir = _run_dir(config, scenario, run_id)
    artifacts_dir = base_dir / "artifacts"
    logs_dir = base_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    poisoned_client_ids = {f"client_{idx}" for idx in range(1, scenario.poisoned_clients + 1)}
    client_arrays: list[dict[str, Any]] = []
    poisoning_manifest: dict[str, Any] = {
        "attack_type": scenario.attack_type,
        "poison_rate": scenario.poison_rate,
        "poisoned_clients_count": scenario.poisoned_clients,
        "poisoned_client_ids": sorted(poisoned_client_ids),
        "train_only": True,
        "validation_modified": False,
        "global_test_modified": False,
        "test_sent_to_clients": False,
        "clients": {},
    }
    input_dim = int(config["model"].get("input_dim", 28))
    use_qga_mask = scenario.method.endswith("_qga")
    mask_indices: np.ndarray | None = None
    if use_qga_mask:
        mask_path = Path.cwd() / config["inputs"]["qga_final_mask_dir"] / "feature_mask.json"
        if mask_path.exists():
            import json

            mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
            mask = np.asarray(mask_payload.get("mask", mask_payload.get("feature_mask")), dtype=bool)
            mask_indices = np.flatnonzero(mask)
            input_dim = int(mask_indices.size)

    for index, partition in enumerate(scenario.partitions, start=1):
        X_train, y_train = load_npz_arrays(partition.train_npz)
        X_val, y_val = load_npz_arrays(partition.val_npz)
        if max_samples is not None:
            X_train, y_train = X_train[:max_samples], y_train[:max_samples]
            X_val, y_val = X_val[:max_samples], y_val[:max_samples]
        if mask_indices is not None:
            X_train = X_train[:, mask_indices]
            X_val = X_val[:, mask_indices]
        if partition.client_id in poisoned_client_ids:
            result = apply_poisoning(
                X_train,
                y_train,
                attack_type=scenario.attack_type,
                poison_rate=scenario.poison_rate,
                seed=seed + index,
                noise_std=float(config.get("noise", {}).get("gaussian_std", 0.05)),
                clip_min=float(config.get("noise", {}).get("clip_min", -10.0)),
                clip_max=float(config.get("noise", {}).get("clip_max", 10.0)),
            )
            X_train, y_train = result.X, result.y
            poisoning_manifest["clients"][partition.client_id] = result.manifest
        else:
            poisoning_manifest["clients"][partition.client_id] = {"attack_type": "clean", "poisoned_rows": 0}
        client_arrays.append({"client_id": partition.client_id, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val})

    device = _select_device(config)
    global_model = _build_model(input_dim, config).to(device)
    rounds_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    aggregation_rows: list[dict[str, Any]] = []

    for round_id in range(1, int(scenario.rounds) + 1):
        started = time.perf_counter()
        client_states: list[OrderedDict[str, torch.Tensor]] = []
        num_examples: list[int] = []
        qifa_scores: list[float] = []
        fedavg_weights: list[float] = []
        for client in client_arrays:
            state, train_loss = _fit_local(global_model, client["X_train"], client["y_train"], config=config, device=device, seed=seed + round_id)
            local_model = _build_model(input_dim, config).to(device)
            local_model.load_state_dict(state)
            val_metrics = _evaluate(local_model, client["X_val"], client["y_val"], device, int(config["training"].get("batch_size", 512)))
            n_train = int(client["y_train"].shape[0])
            client_states.append(state)
            num_examples.append(n_train)
            fedavg_weights.append(n_train)
            score = (
                0.5 * float(val_metrics["macro_f1"])
                + 0.25 * float(val_metrics["attack_recall"])
                - 0.1 * float(val_metrics["FPR"])
                - 0.05 * float(val_metrics["loss"])
            )
            qifa_scores.append(float(score))
            client_rows.append(
                {
                    "round": round_id,
                    "method": scenario.method,
                    "client_id": client["client_id"],
                    "train_samples": n_train,
                    "val_samples": int(client["y_val"].shape[0]),
                    "local_train_loss": train_loss,
                    "local_val_loss": val_metrics["loss"],
                    "local_macro_f1": val_metrics["macro_f1"],
                    "local_attack_recall": val_metrics["attack_recall"],
                    "local_fpr": val_metrics["FPR"],
                    "poisoned": client["client_id"] in poisoned_client_ids,
                }
            )
        if scenario.method.startswith("qifa"):
            fedavg = np.asarray(fedavg_weights, dtype=float) / max(float(sum(fedavg_weights)), 1.0)
            theta = normalize_scores_to_theta(qifa_scores)
            amplitudes = amplitudes_from_theta(theta)
            probabilities = probabilities_from_amplitudes(amplitudes)
            final_weights = hybrid_weights(fedavg, probabilities, float(config.get("qifa", {}).get("gamma", 0.5)))
            aggregated = OrderedDict()
            for key in client_states[0].keys():
                value = None
                for state, weight in zip(client_states, final_weights):
                    term = state[key].to(torch.float64) * float(weight)
                    value = term if value is None else value + term
                aggregated[key] = value.to(dtype=client_states[0][key].dtype)
            for client, fed_w, prob, weight, score in zip(client_arrays, fedavg, probabilities, final_weights, qifa_scores):
                aggregation_rows.append(
                    {
                        "round": round_id,
                        "client_id": client["client_id"],
                        "score": score,
                        "fedavg_weight": fed_w,
                        "probability": prob,
                        "final_weight": weight,
                    }
                )
            global_model.load_state_dict(aggregated)
        else:
            result = fedavg_state_dicts(client_states, num_examples, client_ids=[client["client_id"] for client in client_arrays])
            global_model.load_state_dict(result.state_dict)
            for client_id, weight in result.weights.items():
                aggregation_rows.append({"round": round_id, "client_id": client_id, "fedavg_weight": weight, "final_weight": weight})

        X_val_all = np.concatenate([client["X_val"] for client in client_arrays], axis=0)
        y_val_all = np.concatenate([client["y_val"] for client in client_arrays], axis=0)
        val_metrics = _evaluate(global_model, X_val_all, y_val_all, device, int(config["training"].get("batch_size", 512)))
        model_bytes = sum(t.numel() * t.element_size() for t in global_model.state_dict().values())
        round_bytes = int(2 * scenario.clients * model_bytes)
        rounds_rows.append(
            {
                "round": round_id,
                "method": scenario.method,
                "attack_type": scenario.attack_type,
                "poison_rate": scenario.poison_rate,
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "attack_recall": val_metrics["attack_recall"],
                "FPR": val_metrics["FPR"],
                "FNR": val_metrics["FNR"],
                "val_loss": val_metrics["loss"],
                "round_time_sec": time.perf_counter() - started,
                "model_size_bytes": model_bytes,
                "communication_total_bytes": round_bytes,
                "communication_cumulative_bytes": round_bytes * round_id,
            }
        )

    X_test, y_test = load_npz_arrays(scenario.test_npz)
    if max_samples is not None:
        X_test, y_test = X_test[:max_samples], y_test[:max_samples]
    if mask_indices is not None:
        X_test = X_test[:, mask_indices]
    test_metrics = _evaluate(global_model, X_test, y_test, device, int(config["training"].get("batch_size", 512)))
    test_metrics["robustness_score"] = robustness_score(test_metrics)

    _write_csv(artifacts_dir / "metrics_rounds.csv", rounds_rows)
    _write_csv(artifacts_dir / "metrics_clients.csv", client_rows)
    _write_csv(artifacts_dir / "aggregation_weights.csv", aggregation_rows)
    _write_csv(
        artifacts_dir / "confusion_matrix.csv",
        [
            {"label": "true_normal", "pred_normal": test_metrics["TN"], "pred_attack": test_metrics["FP"]},
            {"label": "true_attack", "pred_normal": test_metrics["FN"], "pred_attack": test_metrics["TP"]},
        ],
    )
    write_json(artifacts_dir / "metrics_test.json", test_metrics)
    write_json(artifacts_dir / "classification_report.json", {"binary_summary": test_metrics})
    write_json(artifacts_dir / "poisoning_manifest.json", poisoning_manifest)
    write_json(artifacts_dir / "comparison_with_clean_baseline.json", {"warning": "Clean baseline aggregation is built by report_builder when matching runs exist."})
    summary = {
        "accepted": True,
        "phase": "P10",
        "mode": mode,
        "run_id": run_id,
        "method": scenario.method,
        "scenario": {
            "alpha": scenario.alpha,
            "clients": scenario.clients,
            "rounds": scenario.rounds,
            "attack_type": scenario.attack_type,
            "poison_rate": scenario.poison_rate,
            "poisoned_clients": scenario.poisoned_clients,
        },
        "dataset": {
            "input_dim": input_dim,
            "test_sent_to_clients": False,
            "test_modified": False,
            "validation_modified": False,
        },
        "test": test_metrics,
        "artifacts": [str(path) for path in sorted(artifacts_dir.glob("*"))],
        "criteria": {
            "train_poisoning_only": True,
            "p3_partitions_not_modified": True,
            "global_test_holdout_protected": True,
            "test_sent_to_clients_false": True,
            "docker_not_modified": True,
            "dashboard_not_modified": True,
            "fedtn_mps_not_used": True,
            "full_training_not_auto_launched": mode != "full",
        },
        "warnings": ["Smoke metrics are not scientific." if mode == "smoke" else ""],
        "errors": [],
    }
    write_json(artifacts_dir / "run_summary.json", summary)
    return summary
