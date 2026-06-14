"""HeteroFL L1 adapter that applies a frozen QGA feature mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from multitier_heterofl.aggregation import aggregate_slice_weighted
from multitier_heterofl.slicing import extract_tier_state, load_tier_state
from multitier_heterofl.supernet import build_supernet, build_tier_model, model_size_bytes
from qga.config import make_run_id, relative_to_repo, repo_path, write_json
from qga.data import load_l1_npz, sample_arrays
from qga.fedavg_adapter import _adapter_summary, _concat_arrays, _evaluate, _load_client_arrays, _train_one_epoch
from qga.feature_mask import load_latest_mask
from qga.metrics import classification_report_dict, confusion_matrix_rows, predictions_from_threshold
from qga.plotting import plot_binary_adapter_figures
from qga.report_builder import write_csv


TIER_MAPPING = {
    3: {"client_1": "weak", "client_2": "medium", "client_3": "powerful"},
    4: {"client_1": "weak", "client_2": "weak", "client_3": "medium", "client_4": "powerful"},
    5: {"client_1": "weak", "client_2": "weak", "client_3": "medium", "client_4": "medium", "client_5": "powerful"},
}


def run_qga_heterofl_l1(
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
    tier_mapping = TIER_MAPPING[int(clients)]
    run_id = make_run_id()
    base_dir = repo_path(config, "outputs.qga_heterofl_dir") / f"alpha_{alpha}" / f"k{clients}" / "runs" / run_id
    artifacts_dir = base_dir / "artifacts"
    checkpoints_dir = base_dir / "checkpoints"
    figures_dir = repo_path(config, "outputs.figures_dir") / "heterofl_l1" / run_id
    for path in [artifacts_dir, checkpoints_dir, figures_dir, base_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    scenario_dir = repo_path(config, "inputs.l1_partitions_root") / f"alpha_{alpha}" / f"k{clients}"
    client_ids = [f"client_{idx}" for idx in range(1, int(clients) + 1)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supernet = build_supernet(input_dim=selected_count, output_dim=2).to(device)
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
    bandwidth_rows: list[dict[str, Any]] = []
    slices_rows: list[dict[str, Any]] = []
    cumulative_bytes = 0
    best_macro = float("-inf")
    best_state = None
    tier_sizes = {
        tier: model_size_bytes(build_tier_model(tier=tier, input_dim=selected_count, output_dim=2))
        for tier in ["weak", "medium", "powerful"]
    }
    for round_idx in range(1, int(rounds) + 1):
        updates = []
        losses = []
        upload_bytes = 0
        download_bytes = 0
        for client_id, (train_arrays, val_arrays) in client_data.items():
            tier = tier_mapping[client_id]
            model = build_tier_model(tier=tier, input_dim=selected_count, output_dim=2).to(device)
            state = extract_tier_state(supernet.state_dict(), tier)
            load_tier_state(model, state)
            loss = _train_one_epoch(model, train_arrays, mask, batch_size=512, seed=42 + round_idx, device=device)
            val_metrics = _evaluate(model, val_arrays, mask, batch_size=512, seed=42, device=device)
            updates.append(
                {
                    "client_id": client_id,
                    "tier": tier,
                    "num_examples": train_arrays.rows,
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
            )
            losses.append(loss)
            upload_bytes += tier_sizes[tier]
            download_bytes += tier_sizes[tier]
            metrics_clients.append(
                {
                    "round": round_idx,
                    "client_id": client_id,
                    "tier": tier,
                    "train_samples": train_arrays.rows,
                    "val_samples": val_arrays.rows,
                    "local_loss": loss,
                    "local_macro_f1": val_metrics["macro_f1"],
                    "local_attack_recall": val_metrics["recall_attack"],
                    "local_fpr": val_metrics["FPR"],
                    "upload_bytes": tier_sizes[tier],
                    "download_bytes": tier_sizes[tier],
                }
            )
        new_state, slice_info = aggregate_slice_weighted(updates, supernet.state_dict())
        supernet.load_state_dict(new_state)
        val_all = _concat_arrays([pair[1] for pair in client_data.values()])
        val_metrics = _evaluate(supernet, val_all, mask, batch_size=512, seed=42, device=device)
        cumulative_bytes += upload_bytes + download_bytes
        metrics_rounds.append(
            {
                "round": round_idx,
                "alpha": alpha,
                "clients": clients,
                "train_loss_mean": sum(losses) / max(len(losses), 1),
                "val_loss_mean": val_metrics["loss"],
                "macro_f1": val_metrics["macro_f1"],
                "attack_recall": val_metrics["recall_attack"],
                "FPR": val_metrics["FPR"],
                "FNR": val_metrics["FNR"],
                "TP": val_metrics["TP"],
                "TN": val_metrics["TN"],
                "FP": val_metrics["FP"],
                "FN": val_metrics["FN"],
                "communication_total_bytes": upload_bytes + download_bytes,
                "communication_cumulative_bytes": cumulative_bytes,
                "slices_updated_ratio": slice_info["updated_ratio"],
            }
        )
        slices_rows.append({"round": round_idx, **{k: str(v) for k, v in slice_info.items()}})
        bandwidth_rows.append(
            {
                "round": round_idx,
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "total_bytes": upload_bytes + download_bytes,
                "cumulative_bytes": cumulative_bytes,
            }
        )
        if val_metrics["macro_f1"] > best_macro:
            best_macro = float(val_metrics["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in supernet.state_dict().items()}
    if best_state is not None:
        torch.save(best_state, checkpoints_dir / "best_global_supernet.pth")
    torch.save(supernet.state_dict(), checkpoints_dir / "last_global_supernet.pth")
    test_arrays = load_l1_npz(repo_path(config, "inputs.test_npz"))
    if mode == "smoke":
        test_arrays = sample_arrays(test_arrays, max_samples=max_samples_per_client or 1000, seed=99)
    test_metrics = _evaluate(supernet, test_arrays, mask, batch_size=512, seed=42, device=device)
    y_pred = predictions_from_threshold(test_metrics["prob_attack"], 0.5)
    test_metrics_clean = {key: value for key, value in test_metrics.items() if key != "prob_attack"}
    write_csv(artifacts_dir / "metrics_rounds.csv", metrics_rounds)
    write_csv(artifacts_dir / "metrics_clients.csv", metrics_clients)
    write_csv(artifacts_dir / "bandwidth_by_tier.csv", bandwidth_rows)
    write_csv(artifacts_dir / "slices_updated.csv", slices_rows)
    write_json(artifacts_dir / "metrics_test.json", test_metrics_clean)
    write_json(artifacts_dir / "metrics_val.json", metrics_rounds[-1])
    write_json(artifacts_dir / "classification_report.json", classification_report_dict(test_arrays.y, y_pred))
    write_csv(artifacts_dir / "confusion_matrix.csv", confusion_matrix_rows(test_metrics_clean))
    write_json(artifacts_dir / "tier_mapping.json", tier_mapping)
    write_json(
        artifacts_dir / "tier_model_configs.json",
        {
            "weak": {"input_dim": selected_count, "hidden_layers": [64], "output_dim": 2},
            "medium": {"input_dim": selected_count, "hidden_layers": [128, 64], "output_dim": 2},
            "powerful": {"input_dim": selected_count, "hidden_layers": [256, 128], "output_dim": 2},
        },
    )
    write_json(
        artifacts_dir / "model_config.json",
        {"name": "HeteroFL_L1_Supernet+QGA", "input_dim": selected_count, "hidden_layers": [256, 128], "output_dim": 2},
    )
    write_json(artifacts_dir / "selected_features_reference.json", mask_info["payload"])
    comparison = {"p8_macro_f1": test_metrics_clean.get("macro_f1"), "p7_reference": config["inputs"].get("p7_multitier_summary")}
    write_json(artifacts_dir / "comparison_with_p7.json", comparison)
    figures = plot_binary_adapter_figures(
        metrics_rounds=metrics_rounds,
        confusion_metrics=test_metrics_clean,
        output_dir=figures_dir,
        prefix="p8_heterofl_qga",
    )
    run_summary_path = artifacts_dir / "run_summary.json"
    artifacts = [relative_to_repo(path, config) for path in artifacts_dir.iterdir() if path.is_file()]
    artifacts.append(relative_to_repo(run_summary_path, config))
    summary = _adapter_summary(
        config=config,
        task="qga_heterofl_l1",
        method="HeteroFL + QGA",
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
        model_size_bytes=model_size_bytes(supernet),
        cumulative_bytes=cumulative_bytes,
    )
    write_json(run_summary_path, summary)
    write_json(base_dir.parent.parent / "latest_run_summary.json", summary)
    return summary
