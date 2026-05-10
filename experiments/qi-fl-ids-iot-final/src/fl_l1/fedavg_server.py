"""In-process FedAvg server orchestration for P5 L1."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from models.l1_mlp import CentralizedL1MLP

from .aggregation import fedavg_state_dicts
from .client_data import load_client_npz, make_dataloader
from .communication import model_size_bytes, round_bandwidth
from .evaluation import evaluate_loader, finalize_test_metrics, tune_threshold_on_validation
from .fedavg_client import FedAvgL1Client
from .round_logger import (
    ConsoleLogger,
    RoundLogger,
    format_client_console_line,
    format_round_console_line,
)
from .scenario_loader import L1Scenario, rel, write_json


def select_device(device_config: str) -> torch.device:
    """Resolve configured device."""

    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def build_model(config: dict[str, Any]) -> CentralizedL1MLP:
    """Build the P5 L1 MLP, architecture-matched with P4."""

    model_cfg = config["model"]
    return CentralizedL1MLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_layers=list(model_cfg["hidden_layers"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
    )


def _alpha_run_name(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def resolve_sample_limit_for_mode(
    *,
    mode: str,
    requested_max_samples: int | None,
    default_smoke_max_samples: int | None,
) -> int | None:
    """Apply max-samples only in smoke mode.

    Full and grid runs must always use every client train/val sample from P3.
    """

    if mode != "smoke":
        return None
    if requested_max_samples is not None:
        return int(requested_max_samples)
    if default_smoke_max_samples is not None:
        return int(default_smoke_max_samples)
    return None


def _scenario_run_dir(config: dict[str, Any], repo_root: Path, scenario: L1Scenario) -> Path:
    return repo_root / config["outputs"]["run_dir"] / _alpha_run_name(scenario.alpha) / f"k{scenario.num_clients}"


def _scenario_figures_dir(config: dict[str, Any], repo_root: Path, scenario: L1Scenario) -> Path:
    return (
        repo_root
        / config["outputs"]["figures_fl_dir"]
        / _alpha_run_name(scenario.alpha)
        / f"k{scenario.num_clients}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_rounds(artifacts_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics_rounds.csv"
    bandwidth_path = artifacts_dir / "bandwidth_rounds.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return []
    generated: list[Path] = []
    rounds = [int(row["round"]) for row in rows]

    plot_specs = [
        ("macro_f1", "fl_l1_macro_f1_by_round.png", "Macro-F1 by round", "#2563EB"),
        ("train_loss_mean", "fl_l1_loss_by_round.png", "Train loss by round", "#DC2626"),
        ("FPR", "fl_l1_fpr_by_round.png", "FPR by round", "#F97316"),
    ]
    for key, filename, title, color in plot_specs:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, [float(row[key]) for row in rows], marker="o", color=color)
        ax.set_xlabel("Round")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    if bandwidth_path.exists():
        with bandwidth_path.open("r", encoding="utf-8", newline="") as file:
            bw_rows = list(csv.DictReader(file))
        if bw_rows:
            path = figures_dir / "fl_l1_bandwidth_by_round.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                [int(row["round"]) for row in bw_rows],
                [float(row["cumulative_mb"]) for row in bw_rows],
                marker="o",
                color="#16A34A",
            )
            ax.set_xlabel("Round")
            ax.set_ylabel("Cumulative MB")
            ax.set_title("FedAvg bandwidth by round")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            generated.append(path)
    return generated


def _write_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    _write_csv(
        path,
        [
            {
                "label": "true_normal",
                "pred_normal": int(metrics["TN"]),
                "pred_attack": int(metrics["FP"]),
            },
            {
                "label": "true_attack",
                "pred_normal": int(metrics["FN"]),
                "pred_attack": int(metrics["TP"]),
            },
        ],
    )


def _plot_confusion_matrix(metrics: dict[str, Any], path: Path) -> Path:
    matrix = np.asarray([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred normal", "pred attack"])
    ax.set_yticks([0, 1], labels=["true normal", "true attack"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    ax.set_title("P5 FedAvg L1 confusion matrix")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_vs_p4(comparison: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["accuracy", "macro_f1", "attack_recall", "fpr"]
    p4_values = [comparison[f"p4_{label}"] for label in labels]
    p5_values = [comparison[f"p5_{label}"] for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, p4_values, 0.36, label="P4 centralized", color="#64748B")
    ax.bar(x + 0.18, p5_values, 0.36, label="P5 FedAvg", color="#2563EB")
    ax.set_xticks(x, labels=labels, rotation=20, ha="right")
    ax.set_title("P5 FedAvg vs P4 centralized")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_fedavg_scenario(
    *,
    config: dict[str, Any],
    repo_root: Path,
    scenario: L1Scenario,
    mode: str,
    rounds: int,
    max_samples_per_client: int | None = None,
) -> dict[str, Any]:
    """Run one local FedAvg scenario.

    This is intentionally in-process and deterministic; it mirrors FedAvg logic
    without depending on networked Flower execution.
    """

    training_cfg = config["training"]
    threshold_cfg = config["threshold"]
    logging_cfg = config.get("logging", {})
    sample_limit = resolve_sample_limit_for_mode(
        mode=mode,
        requested_max_samples=max_samples_per_client,
        default_smoke_max_samples=config.get("execution", {}).get("smoke_max_samples_per_client"),
    )
    run_dir = _scenario_run_dir(config, repo_root, scenario).resolve()
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    for directory in [checkpoints_dir, artifacts_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    console_logger = ConsoleLogger(logs_dir / "run_console.log", reset=True)
    verbose_rounds = bool(logging_cfg.get("verbose_rounds", True))
    verbose_clients = bool(logging_cfg.get("verbose_clients", False))
    flower_like_logs = bool(logging_cfg.get("flower_like_logs", True))
    log_every_round = max(1, int(logging_cfg.get("log_every_round", 1)))

    def flower_log(message: str) -> None:
        if flower_like_logs:
            console_logger.log(message)

    flower_log("Starting FedAvg L1 server")
    flower_log(f"Loading scenario alpha={scenario.alpha} K={scenario.num_clients}")
    if mode == "smoke":
        console_logger.log(f"Smoke mode enabled: max_samples_per_client={sample_limit}")
    else:
        console_logger.log("Full/grid sample policy: max_samples_per_client ignored; using all client samples")

    device = select_device(str(training_cfg["device"]))
    torch.manual_seed(int(training_cfg["seed"]))
    model_factory = lambda: build_model(config)
    global_model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()

    clients: list[FedAvgL1Client] = []
    client_sample_audit: list[dict[str, Any]] = []
    for index, client_partition in enumerate(scenario.clients, start=1):
        train_arrays = load_client_npz(
            client_partition.train_npz,
            max_samples=sample_limit,
            seed=int(training_cfg["seed"]) + index,
        )
        val_arrays = load_client_npz(
            client_partition.val_npz,
            max_samples=sample_limit,
            seed=int(training_cfg["seed"]) + 10_000 + index,
        )
        train_uses_all = train_arrays.num_samples == client_partition.train_samples
        val_uses_all = val_arrays.num_samples == client_partition.val_samples
        client_sample_audit.append(
            {
                "client_id": client_partition.client_id,
                "expected_train_samples": client_partition.train_samples,
                "loaded_train_samples": train_arrays.num_samples,
                "expected_val_samples": client_partition.val_samples,
                "loaded_val_samples": val_arrays.num_samples,
                "train_uses_all_samples": train_uses_all,
                "val_uses_all_samples": val_uses_all,
            }
        )
        flower_log(
            "Registering "
            f"{client_partition.client_id} train={train_arrays.num_samples}/"
            f"{client_partition.train_samples} val={val_arrays.num_samples}/"
            f"{client_partition.val_samples}"
        )
        clients.append(
            FedAvgL1Client(
                client_id=client_partition.client_id,
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                model_factory=model_factory,
                batch_size=int(training_cfg["batch_size"]),
                learning_rate=float(training_cfg["learning_rate"]),
                weight_decay=float(training_cfg["weight_decay"]),
                local_epochs=int(config["federated"]["local_epochs"]),
                device=device,
                seed=int(training_cfg["seed"]),
            )
        )

    full_uses_all_client_samples = (
        mode in {"full", "grid"}
        and sample_limit is None
        and all(item["train_uses_all_samples"] and item["val_uses_all_samples"] for item in client_sample_audit)
    )
    logger = RoundLogger(artifacts_dir=artifacts_dir, logs_dir=logs_dir, reset=True)
    best_macro_f1 = -1.0
    best_round = 0
    cumulative_bytes = 0
    round_rows: list[dict[str, Any]] = []

    for round_number in range(1, int(rounds) + 1):
        flower_log(f"Starting round {round_number}/{int(rounds)}")
        round_start = time.perf_counter()
        global_state = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }
        size_bytes = model_size_bytes(global_state)
        client_results = []
        for client in clients:
            result = client.fit(global_state, round_number=round_number)
            client_results.append(result)
            flower_log(
                "Client fit completed | "
                f"round={round_number} client={result.client_id} train_samples={result.num_examples}"
            )
            if verbose_clients:
                console_logger.log(format_client_console_line(result.metrics))

        aggregation_start = time.perf_counter()
        flower_log("Aggregating updates with FedAvg weighted by num_examples")
        aggregation = fedavg_state_dicts(
            [result.state_dict for result in client_results],
            [result.num_examples for result in client_results],
            client_ids=[result.client_id for result in client_results],
        )
        aggregation_time_sec = time.perf_counter() - aggregation_start
        global_model.load_state_dict(aggregation.state_dict, strict=True)

        for result in client_results:
            logger.log_client(result.metrics)
            logger.log_aggregation_weight(
                {
                    "round": round_number,
                    "client_id": result.client_id,
                    "num_examples": result.num_examples,
                    "aggregation_weight": aggregation.weights[result.client_id],
                }
            )

        val_weights = np.asarray([result.num_examples for result in client_results], dtype=np.float64)
        train_loss_mean = float(
            np.average([result.metrics["local_train_loss"] for result in client_results], weights=val_weights)
        )
        val_loss_mean = float(
            np.average([result.metrics["local_val_loss"] for result in client_results], weights=val_weights)
        )
        flower_log("Evaluating global model on federated validation")
        all_val_y: list[np.ndarray] = []
        all_val_prob: list[np.ndarray] = []
        for client in clients:
            loader = make_dataloader(
                client.val_arrays,
                batch_size=int(training_cfg["batch_size"]) * 4,
                shuffle=False,
                seed=int(training_cfg["seed"]),
                device=device,
            )
            eval_result = evaluate_loader(
                global_model,
                loader,
                criterion,
                device,
                threshold=0.5,
                collect_probabilities=True,
            )
            all_val_y.append(eval_result["y_true"])
            all_val_prob.append(eval_result["prob_attack"])
        val_y = np.concatenate(all_val_y)
        val_prob = np.concatenate(all_val_prob)
        threshold_payload, _ = tune_threshold_on_validation(
            val_y,
            val_prob,
            start=float(threshold_cfg["start"]),
            stop=float(threshold_cfg["stop"]),
            step=float(threshold_cfg["step"]),
        )
        primary_threshold = float(threshold_payload["primary_threshold"])
        from models.metrics import binary_metrics, predictions_from_threshold

        val_metrics = binary_metrics(val_y, predictions_from_threshold(val_prob, primary_threshold))
        bandwidth = round_bandwidth(
            model_size_bytes_value=size_bytes,
            num_clients=scenario.num_clients,
            previous_cumulative_bytes=cumulative_bytes,
        )
        cumulative_bytes = int(bandwidth["cumulative_bytes"])
        round_time_sec = time.perf_counter() - round_start
        round_row = {
            "round": round_number,
            "alpha": scenario.alpha,
            "num_clients": scenario.num_clients,
            "train_loss_mean": train_loss_mean,
            "val_loss_mean": val_loss_mean,
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision_attack"],
            "recall": val_metrics["recall_attack"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "attack_precision": val_metrics["precision_attack"],
            "attack_recall": val_metrics["recall_attack"],
            "attack_f1": val_metrics["f1_attack"],
            "FPR": val_metrics["FPR"],
            "FNR": val_metrics["FNR"],
            "TP": val_metrics["TP"],
            "TN": val_metrics["TN"],
            "FP": val_metrics["FP"],
            "FN": val_metrics["FN"],
            "round_time_sec": round_time_sec,
            "aggregation_time_sec": aggregation_time_sec,
            "model_size_bytes": size_bytes,
            "communication_upload_bytes": bandwidth["upload_bytes"],
            "communication_download_bytes": bandwidth["download_bytes"],
            "communication_total_bytes": bandwidth["total_bytes"],
            "communication_cumulative_bytes": bandwidth["cumulative_bytes"],
        }
        logger.log_round(round_row)
        logger.log_bandwidth({"round": round_number, **bandwidth})
        logger.log_event({"event": "round_completed", **round_row})
        if verbose_rounds and (round_number % log_every_round == 0 or round_number == int(rounds)):
            console_logger.log(
                format_round_console_line(
                    round_row,
                    current_round=round_number,
                    total_rounds=int(rounds),
                )
            )
        round_rows.append(round_row)

        torch.save(
            {
                "model_state_dict": global_model.state_dict(),
                "round": round_number,
                "selection_split": "federated_validation",
                "threshold": primary_threshold,
            },
            checkpoints_dir / "last_global_model.pth",
        )
        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = round_number
            flower_log(f"Saving best checkpoint if improved | improved=true round={round_number}")
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "round": round_number,
                    "selection_metric": "val_macro_f1",
                    "selection_metric_value": best_macro_f1,
                    "selection_split": "federated_validation",
                    "test_used_for_selection": False,
                    "threshold": primary_threshold,
                },
                checkpoints_dir / "best_global_model.pth",
            )
        else:
            flower_log(f"Saving best checkpoint if improved | improved=false round={round_number}")

    best_checkpoint = torch.load(checkpoints_dir / "best_global_model.pth", map_location=device, weights_only=False)
    global_model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    all_val_y = []
    all_val_prob = []
    for client in clients:
        loader = make_dataloader(
            client.val_arrays,
            batch_size=int(training_cfg["batch_size"]) * 4,
            shuffle=False,
            seed=int(training_cfg["seed"]),
            device=device,
        )
        eval_result = evaluate_loader(
            global_model,
            loader,
            criterion,
            device,
            threshold=0.5,
            collect_probabilities=True,
        )
        all_val_y.append(eval_result["y_true"])
        all_val_prob.append(eval_result["prob_attack"])
    val_y = np.concatenate(all_val_y)
    val_prob = np.concatenate(all_val_prob)
    threshold_payload, threshold_rows = tune_threshold_on_validation(
        val_y,
        val_prob,
        start=float(config["threshold"]["start"]),
        stop=float(config["threshold"]["stop"]),
        step=float(config["threshold"]["step"]),
    )
    primary_threshold = float(threshold_payload["primary_threshold"])
    from models.metrics import binary_metrics, predictions_from_threshold

    metrics_val = binary_metrics(val_y, predictions_from_threshold(val_prob, primary_threshold))
    metrics_val["threshold"] = primary_threshold
    metrics_val["selection_split"] = "federated_validation"
    metrics_val["test_used_for_threshold"] = False

    flower_log("Final evaluation on global test holdout")
    test_arrays = load_client_npz(
        scenario.global_test_npz,
        max_samples=sample_limit if mode == "smoke" else None,
        seed=int(training_cfg["seed"]) + 99_000,
    )
    test_loader = make_dataloader(
        test_arrays,
        batch_size=int(training_cfg["batch_size"]) * 4,
        shuffle=False,
        seed=int(training_cfg["seed"]),
        device=device,
    )
    test_eval = evaluate_loader(
        global_model,
        test_loader,
        criterion,
        device,
        threshold=primary_threshold,
        collect_probabilities=True,
    )
    metrics_test = finalize_test_metrics(
        test_eval["y_true"],
        test_eval["prob_attack"],
        primary_threshold,
    )
    metrics_test["threshold"] = primary_threshold
    metrics_test["model_size_bytes"] = model_size_bytes(global_model.state_dict())
    metrics_test["num_parameters"] = sum(
        parameter.numel() for parameter in global_model.parameters() if parameter.requires_grad
    )
    write_json(artifacts_dir / "metrics_val.json", metrics_val)
    write_json(artifacts_dir / "metrics_test.json", metrics_test)
    write_json(artifacts_dir / "threshold.json", threshold_payload)
    _write_csv(artifacts_dir / "threshold_sweep.csv", threshold_rows)
    _write_confusion_matrix(artifacts_dir / "confusion_matrix.csv", metrics_test)
    model_config = {
        "model": config["model"],
        "federated": config["federated"],
        "training": config["training"],
        "data_usage": {
            "p3_l1_train_val_clients": True,
            "global_test_holdout_final_only": True,
            "test_partitioned": False,
            "sample_policy": {
                "mode": mode,
                "requested_max_samples_per_client": max_samples_per_client,
                "effective_max_samples_per_client": sample_limit,
                "max_samples_applied_only_in_smoke": True,
                "full_uses_all_client_samples": full_uses_all_client_samples,
                "client_sample_audit": client_sample_audit,
            },
        },
    }
    write_json(artifacts_dir / "model_config.json", model_config)

    p4_metrics_path = repo_root / config["inputs"]["centralized_l1_metrics"]
    p4_metrics = {}
    if p4_metrics_path.exists():
        import json

        p4_metrics = json.loads(p4_metrics_path.read_text(encoding="utf-8"))
    comparison = {
        "p4_accuracy": float(p4_metrics.get("accuracy", 0.0)),
        "p5_accuracy": float(metrics_test["accuracy"]),
        "gap_accuracy": float(metrics_test["accuracy"]) - float(p4_metrics.get("accuracy", 0.0)),
        "p4_macro_f1": float(p4_metrics.get("macro_f1", 0.0)),
        "p5_macro_f1": float(metrics_test["macro_f1"]),
        "gap_macro_f1": float(metrics_test["macro_f1"]) - float(p4_metrics.get("macro_f1", 0.0)),
        "p4_attack_recall": float(p4_metrics.get("recall_attack", 0.0)),
        "p5_attack_recall": float(metrics_test["recall_attack"]),
        "gap_attack_recall": float(metrics_test["recall_attack"]) - float(p4_metrics.get("recall_attack", 0.0)),
        "p4_fpr": float(p4_metrics.get("FPR", 0.0)),
        "p5_fpr": float(metrics_test["FPR"]),
        "gap_fpr": float(metrics_test["FPR"]) - float(p4_metrics.get("FPR", 0.0)),
    }
    write_json(artifacts_dir / "comparison_with_p4.json", comparison)

    figures_dir = _scenario_figures_dir(config, repo_root, scenario).resolve()
    figures = _plot_rounds(
        artifacts_dir,
        figures_dir,
    )
    figures.append(_plot_confusion_matrix(metrics_test, figures_dir / "fl_l1_confusion_matrix.png"))
    figures.append(_plot_vs_p4(comparison, figures_dir / "fl_l1_vs_centralized.png"))
    run_summary = {
        "mode": mode,
        "dataset_level": "l1_binary",
        "alpha": scenario.alpha,
        "num_clients": scenario.num_clients,
        "rounds": int(rounds),
        "best_round": best_round,
        "best_val_macro_f1": best_macro_f1,
        "threshold": threshold_payload,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "comparison_with_p4": comparison,
        "global_test_holdout": rel(scenario.global_test_npz, repo_root),
        "test_partitioned": False,
        "sample_policy": {
            "mode": mode,
            "requested_max_samples_per_client": max_samples_per_client,
            "effective_max_samples_per_client": sample_limit,
            "max_samples_applied_only_in_smoke": True,
            "full_uses_all_client_samples": full_uses_all_client_samples,
            "client_sample_audit": client_sample_audit,
        },
        "client_count": len(clients),
        "round_rows": round_rows,
        "figures": [rel(path, repo_root) for path in figures],
        "run_console_log": rel(logs_dir / "run_console.log", repo_root),
    }
    write_json(artifacts_dir / "run_summary.json", run_summary)
    manifest = {
        "run_summary": rel(artifacts_dir / "run_summary.json", repo_root),
        "scenario_manifest": rel(scenario.manifest_path, repo_root),
        "global_test_reference": rel(scenario.global_test_reference_path, repo_root),
        "checkpoints_dir": rel(checkpoints_dir, repo_root),
        "artifacts_dir": rel(artifacts_dir, repo_root),
        "logs_dir": rel(logs_dir, repo_root),
        "run_console_log": rel(logs_dir / "run_console.log", repo_root),
        "full_uses_all_client_samples": full_uses_all_client_samples,
        "sample_policy": run_summary["sample_policy"],
    }
    write_json(run_dir / "manifest.json", manifest)
    console_logger.log(
        "FedAvg L1 run complete | "
        f"mode={mode} alpha={scenario.alpha} K={scenario.num_clients} "
        f"best_round={best_round} test_macro_f1={float(metrics_test['macro_f1']):.4f}"
    )
    return run_summary
