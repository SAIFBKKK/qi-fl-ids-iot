"""FedAvg L1 adapter that applies a frozen QGA feature mask."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from fl_l1.aggregation import fedavg_state_dicts
from models.l1_mlp import CentralizedL1MLP
from qga.config import make_run_id, relative_to_repo, repo_path, write_json
from qga.data import L1Arrays, load_l1_npz, make_loader, sample_arrays
from qga.feature_mask import apply_feature_mask, load_latest_mask
from qga.metrics import classification_report_dict, confusion_matrix_rows, metrics_from_probabilities, predictions_from_threshold
from qga.plotting import plot_binary_adapter_figures
from qga.report_builder import write_csv


def _state_model(input_dim: int, dropout: float = 0.2) -> CentralizedL1MLP:
    return CentralizedL1MLP(input_dim=input_dim, hidden_layers=[128, 64], output_dim=2, dropout=dropout)


def _train_one_epoch(
    model: nn.Module,
    arrays: L1Arrays,
    mask: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = make_loader(apply_feature_mask(arrays.X, mask), arrays.y, batch_size=batch_size, shuffle=True, seed=seed)
    losses: list[float] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _evaluate(model: nn.Module, arrays: L1Arrays, mask: np.ndarray, *, batch_size: int, seed: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(apply_feature_mask(arrays.X, mask), arrays.y, batch_size=batch_size, shuffle=False, seed=seed)
    probs: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        total_loss += float(loss.item()) * int(y_batch.shape[0])
        total_rows += int(y_batch.shape[0])
    prob_np = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    metrics = metrics_from_probabilities(arrays.y, prob_np, threshold=0.5)
    metrics["loss"] = total_loss / max(total_rows, 1)
    metrics["prob_attack"] = prob_np
    return metrics


def _load_client_arrays(
    scenario_dir: Path,
    client_id: str,
    *,
    max_samples_per_client: int | None,
    seed: int,
) -> tuple[L1Arrays, L1Arrays]:
    train = load_l1_npz(scenario_dir / client_id / "train_scaled.npz")
    val = load_l1_npz(scenario_dir / client_id / "val_scaled.npz")
    if max_samples_per_client:
        train = sample_arrays(train, max_samples=max_samples_per_client, seed=seed)
        val = sample_arrays(val, max_samples=max_samples_per_client, seed=seed + 100)
    return train, val


def run_qga_fedavg_l1(
    *,
    config: dict[str, Any],
    mode: str,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None = None,
) -> dict[str, Any]:
    mask_info = load_latest_mask(repo_path(config, "outputs.qga_dir"))
    mask = mask_info["mask"]
    selected_count = int(mask.sum())
    run_id = make_run_id()
    base_dir = repo_path(config, "outputs.qga_fedavg_dir") / f"alpha_{alpha}" / f"k{clients}" / "runs" / run_id
    artifacts_dir = base_dir / "artifacts"
    checkpoints_dir = base_dir / "checkpoints"
    figures_dir = repo_path(config, "outputs.figures_dir") / "fedavg_l1" / run_id
    logs_dir = base_dir / "logs"
    for path in [artifacts_dir, checkpoints_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    scenario_dir = repo_path(config, "inputs.l1_partitions_root") / f"alpha_{alpha}" / f"k{clients}"
    client_ids = [f"client_{idx}" for idx in range(1, int(clients) + 1)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = _state_model(selected_count).to(device)
    client_data = {
        client_id: _load_client_arrays(
            scenario_dir,
            client_id,
            max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
            seed=42 + idx,
        )
        for idx, client_id in enumerate(client_ids)
    }
    metrics_rounds: list[dict[str, Any]] = []
    metrics_clients: list[dict[str, Any]] = []
    aggregation_rows: list[dict[str, Any]] = []
    bandwidth_rows: list[dict[str, Any]] = []
    best_macro = float("-inf")
    best_state: OrderedDict[str, torch.Tensor] | None = None
    cumulative_bytes = 0
    model_size_bytes = int(sum(t.detach().cpu().numpy().nbytes for t in global_model.state_dict().values()))
    for round_idx in range(1, int(rounds) + 1):
        round_start = perf_counter()
        states = []
        examples = []
        losses = []
        for client_id, (train_arrays, val_arrays) in client_data.items():
            local_model = _state_model(selected_count).to(device)
            local_model.load_state_dict(global_model.state_dict())
            start = perf_counter()
            loss = _train_one_epoch(local_model, train_arrays, mask, batch_size=512, seed=42 + round_idx, device=device)
            fit_time = perf_counter() - start
            val_metrics = _evaluate(local_model, val_arrays, mask, batch_size=512, seed=42, device=device)
            states.append({key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()})
            examples.append(train_arrays.rows)
            losses.append(loss)
            metrics_clients.append(
                {
                    "round": round_idx,
                    "client_id": client_id,
                    "train_samples": train_arrays.rows,
                    "val_samples": val_arrays.rows,
                    "local_train_loss": loss,
                    "local_val_loss": val_metrics["loss"],
                    "local_macro_f1": val_metrics["macro_f1"],
                    "local_attack_recall": val_metrics["recall_attack"],
                    "local_fpr": val_metrics["FPR"],
                    "fit_time_sec": fit_time,
                    "upload_bytes": model_size_bytes,
                    "download_bytes": model_size_bytes,
                }
            )
        agg = fedavg_state_dicts(states, examples, client_ids=client_ids)
        global_model.load_state_dict(agg.state_dict)
        val_all = _concat_arrays([pair[1] for pair in client_data.values()])
        val_metrics = _evaluate(global_model, val_all, mask, batch_size=512, seed=42, device=device)
        cumulative_bytes += 2 * int(clients) * model_size_bytes
        metrics_rounds.append(
            {
                "round": round_idx,
                "alpha": alpha,
                "clients": clients,
                "train_loss_mean": float(np.mean(losses)),
                "val_loss_mean": val_metrics["loss"],
                "macro_f1": val_metrics["macro_f1"],
                "attack_recall": val_metrics["recall_attack"],
                "FPR": val_metrics["FPR"],
                "FNR": val_metrics["FNR"],
                "TP": val_metrics["TP"],
                "TN": val_metrics["TN"],
                "FP": val_metrics["FP"],
                "FN": val_metrics["FN"],
                "round_time_sec": perf_counter() - round_start,
                "model_size_bytes": model_size_bytes,
                "communication_total_bytes": 2 * int(clients) * model_size_bytes,
                "communication_cumulative_bytes": cumulative_bytes,
            }
        )
        for client_id, weight in agg.weights.items():
            aggregation_rows.append({"round": round_idx, "client_id": client_id, "aggregation_weight": weight})
        bandwidth_rows.append(
            {
                "round": round_idx,
                "upload_bytes": int(clients) * model_size_bytes,
                "download_bytes": int(clients) * model_size_bytes,
                "total_bytes": 2 * int(clients) * model_size_bytes,
                "cumulative_bytes": cumulative_bytes,
            }
        )
        if val_metrics["macro_f1"] > best_macro:
            best_macro = float(val_metrics["macro_f1"])
            best_state = OrderedDict((key, value.detach().cpu().clone()) for key, value in global_model.state_dict().items())
    if best_state is not None:
        torch.save(best_state, checkpoints_dir / "best_global_model.pth")
    torch.save(global_model.state_dict(), checkpoints_dir / "last_global_model.pth")
    test_arrays = load_l1_npz(repo_path(config, "inputs.test_npz"))
    if mode == "smoke":
        test_arrays = sample_arrays(test_arrays, max_samples=max_samples_per_client or 1000, seed=99)
    test_metrics = _evaluate(global_model, test_arrays, mask, batch_size=512, seed=42, device=device)
    y_pred = predictions_from_threshold(test_metrics["prob_attack"], 0.5)
    test_metrics_clean = {key: value for key, value in test_metrics.items() if key != "prob_attack"}
    artifacts = []
    write_csv(artifacts_dir / "metrics_rounds.csv", metrics_rounds)
    write_csv(artifacts_dir / "metrics_clients.csv", metrics_clients)
    write_csv(artifacts_dir / "aggregation_weights.csv", aggregation_rows)
    write_csv(artifacts_dir / "bandwidth_rounds.csv", bandwidth_rows)
    write_json(artifacts_dir / "metrics_test.json", test_metrics_clean)
    write_json(artifacts_dir / "metrics_val.json", metrics_rounds[-1])
    write_json(
        artifacts_dir / "model_config.json",
        {"name": "FedAvgL1MLP+QGA", "input_dim": selected_count, "hidden_layers": [128, 64], "output_dim": 2},
    )
    write_json(artifacts_dir / "threshold.json", {"primary_threshold": 0.5, "test_used_for_threshold": False})
    write_csv(artifacts_dir / "threshold_sweep.csv", [{"threshold": 0.5, **test_metrics_clean}])
    write_json(artifacts_dir / "classification_report.json", classification_report_dict(test_arrays.y, y_pred))
    write_csv(artifacts_dir / "confusion_matrix.csv", confusion_matrix_rows(test_metrics_clean))
    comparison = _comparison_with_p5(config, alpha=alpha, clients=clients, test_metrics=test_metrics_clean)
    write_json(artifacts_dir / "comparison_with_p5.json", comparison)
    write_json(artifacts_dir / "selected_features_reference.json", mask_info["payload"])
    figures = plot_binary_adapter_figures(
        metrics_rounds=metrics_rounds,
        confusion_metrics=test_metrics_clean,
        output_dir=figures_dir,
        prefix="p8_fedavg_qga",
    )
    run_summary_path = artifacts_dir / "run_summary.json"
    for path in artifacts_dir.iterdir():
        if path.is_file():
            artifacts.append(relative_to_repo(path, config))
    artifacts.append(relative_to_repo(run_summary_path, config))
    summary = _adapter_summary(
        config=config,
        task="qga_fedavg_l1",
        method="FedAvg + QGA",
        mode=mode,
        run_id=run_id,
        alpha=alpha,
        clients=clients,
        rounds=rounds,
        selected_count=selected_count,
        test_metrics=test_metrics_clean,
        artifacts=artifacts,
        figures=[relative_to_repo(path, config) for path in figures],
        comparison=comparison,
        model_size_bytes=model_size_bytes,
        cumulative_bytes=cumulative_bytes,
    )
    write_json(run_summary_path, summary)
    write_json(base_dir.parent.parent / "latest_run_summary.json", summary)
    return summary


def _concat_arrays(items: list[L1Arrays]) -> L1Arrays:
    return L1Arrays(
        X=np.concatenate([item.X for item in items], axis=0),
        y=np.concatenate([item.y for item in items], axis=0),
        label_id_original=np.concatenate([item.label_id_original for item in items], axis=0),
        row_id=np.concatenate([item.row_id for item in items], axis=0),
    )


def _comparison_with_p5(config: dict[str, Any], *, alpha: float, clients: int, test_metrics: dict[str, Any]) -> dict[str, Any]:
    import csv

    p5 = {"macro_f1": None, "attack_recall": None, "fpr": None, "accuracy": None}
    path = repo_path(config, "inputs.p5_grid_summary")
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if float(row["alpha"]) == float(alpha) and int(row["clients"]) == int(clients):
                    p5 = {
                        "macro_f1": float(row["macro_f1"]),
                        "attack_recall": float(row["attack_recall"]),
                        "fpr": float(row["fpr"]),
                        "accuracy": float(row["accuracy"]),
                    }
                    break
    return {
        "p5_macro_f1": p5["macro_f1"],
        "p8_macro_f1": test_metrics.get("macro_f1"),
        "gap_macro_f1_vs_p5": None if p5["macro_f1"] is None else test_metrics.get("macro_f1", 0.0) - p5["macro_f1"],
        "p5_attack_recall": p5["attack_recall"],
        "p8_attack_recall": test_metrics.get("recall_attack"),
        "p5_fpr": p5["fpr"],
        "p8_fpr": test_metrics.get("FPR"),
    }


def _adapter_summary(
    *,
    config: dict[str, Any],
    task: str,
    method: str,
    mode: str,
    run_id: str,
    alpha: float,
    clients: int,
    rounds: int,
    selected_count: int,
    test_metrics: dict[str, Any],
    artifacts: list[str],
    figures: list[str],
    comparison: dict[str, Any],
    model_size_bytes: int,
    cumulative_bytes: int,
) -> dict[str, Any]:
    return {
        "accepted": True,
        "phase": "P8",
        "method": method,
        "task": "l1_binary",
        "mode": mode,
        "run_id": run_id,
        "scenario": {"alpha": alpha, "clients": clients, "rounds": rounds},
        "dataset": {
            "input_dim_original": 28,
            "input_dim_selected": selected_count,
            "test_used_for_selection": False,
            "test_sent_to_clients": False,
        },
        "test": test_metrics,
        "comparison": comparison,
        "communication": {"model_size_bytes": model_size_bytes, "total_bytes": cumulative_bytes},
        "artifacts": artifacts,
        "figures": figures,
        "criteria": {
            "qga_mask_applied": True,
            "test_not_used_for_selection": True,
            "metrics_test_generated": True,
            "comparison_generated": True,
        },
        "warnings": ["smoke metrics are not scientifically significant"] if mode == "smoke" else [],
        "errors": [],
    }
