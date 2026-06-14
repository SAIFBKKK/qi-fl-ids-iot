"""Runtime paths for P7 HeteroFL."""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from multitier_heterofl.config import alpha_dir, normalize_task, repo_path


@dataclass(frozen=True)
class P7RunPaths:
    run_id: str
    scenario_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    logs_dir: Path


def make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def scenario_dir(config: dict[str, Any], repo_root: Path, *, task: str, alpha: float, clients: int) -> Path:
    return repo_path(repo_root, config["outputs"]["run_dir"]) / normalize_task(task) / alpha_dir(alpha) / f"k{clients}"


def prepare_run_paths(
    config: dict[str, Any],
    repo_root: Path,
    *,
    task: str,
    alpha: float,
    clients: int,
    run_id: str | None = None,
) -> P7RunPaths:
    rid = run_id or make_run_id()
    scen_dir = scenario_dir(config, repo_root, task=task, alpha=alpha, clients=clients)
    run_dir = scen_dir / "runs" / rid
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    for path in [checkpoints_dir, artifacts_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    (scen_dir / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "run_dir": run_dir.relative_to(repo_root).as_posix(),
                "latest_run_summary": (scen_dir / "latest_run_summary.json").relative_to(repo_root).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return P7RunPaths(rid, scen_dir, run_dir, checkpoints_dir, artifacts_dir, logs_dir)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} | {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _confusion_rows(matrix: list[list[int]], labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(matrix):
        payload = {"true_class": labels[index]}
        payload.update({f"pred_{label}": int(value) for label, value in zip(labels, row)})
        rows.append(payload)
    return rows


def _comparison(config: dict[str, Any], repo_root: Path, task: str, alpha: float, clients: int, metrics_test: dict[str, Any]) -> dict[str, Any]:
    from multitier_heterofl.config import load_json

    comparison: dict[str, Any] = {"task": task}
    p4_path = repo_path(repo_root, config["inputs"]["p4_l1_metrics"])
    if p4_path.exists():
        p4 = load_json(p4_path)
        comparison["with_p4"] = {
            "p4_accuracy": float(p4.get("accuracy", 0.0)),
            "p4_macro_f1": float(p4.get("macro_f1", 0.0)),
            "p7_accuracy": float(metrics_test.get("accuracy", 0.0)),
            "p7_macro_f1": float(metrics_test.get("macro_f1", 0.0)),
            "gap_macro_f1": float(metrics_test.get("macro_f1", 0.0)) - float(p4.get("macro_f1", 0.0)),
        }
        comparison["p4_macro_f1"] = float(p4.get("macro_f1", 0.0))
    p5_path = repo_path(repo_root, config["inputs"]["p5_grid_summary"])
    if p5_path.exists():
        with p5_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        matched = next((row for row in rows if float(row["alpha"]) == float(alpha) and int(row["clients"]) == int(clients)), None)
        if matched:
            comparison["with_p5"] = {
                "p5_macro_f1": float(matched.get("macro_f1", 0.0)),
                "p5_fpr": float(matched.get("fpr", 0.0)),
                "p7_macro_f1": float(metrics_test.get("macro_f1", 0.0)),
                "p7_fpr": float(metrics_test.get("FPR", metrics_test.get("FPR_macro", 0.0))),
                "gap_macro_f1": float(metrics_test.get("macro_f1", 0.0)) - float(matched.get("macro_f1", 0.0)),
            }
            comparison["p5_macro_f1"] = float(matched.get("macro_f1", 0.0))
    p6_path = repo_path(repo_root, config["inputs"]["p6_hierarchical_summary"])
    if p6_path.exists():
        p6 = load_json(p6_path)
        l2_summaries = [item for item in p6.get("summaries", []) if item.get("task") == "l2_family"]
        if l2_summaries:
            p6_macro = float(l2_summaries[-1]["test"]["metrics"].get("macro_f1", 0.0))
            comparison["with_p6"] = {
                "p6_l2_macro_f1": p6_macro,
                "p7_macro_f1": float(metrics_test.get("macro_f1", 0.0)),
                "gap_macro_f1": float(metrics_test.get("macro_f1", 0.0)) - p6_macro,
            }
            comparison["p6_macro_f1"] = p6_macro
    return comparison


def run_multitier_heterofl(
    *,
    config: dict[str, Any],
    repo_root: Path,
    task: str,
    mode: str,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    import numpy as np

    from multitier_heterofl.aggregation import aggregate_slice_weighted
    from multitier_heterofl.communication import round_bandwidth_by_tier
    from multitier_heterofl.config import rel, tier_mapping_for_k, write_json
    from multitier_heterofl.data import load_client_data, load_global_test, load_scenario, load_validation_union, task_spec
    from multitier_heterofl.evaluation import evaluate_model
    from multitier_heterofl.plotting import generate_figures
    from multitier_heterofl.slicing import extract_tier_state, load_tier_state
    from multitier_heterofl.summary_schema import criteria, existing_relative, run_artifact_paths, run_figure_paths
    from multitier_heterofl.supernet import architecture_for_tier, build_supernet, build_tier_model, model_size_bytes, tier_parameter_summary
    from multitier_heterofl.training import select_device, train_local

    normalized = normalize_task(task)
    spec = task_spec(config, repo_root, normalized)
    scenario = load_scenario(config, repo_root, task=normalized, alpha=alpha, clients=clients)
    mapping = tier_mapping_for_k(config, clients)
    paths = prepare_run_paths(config, repo_root, task=normalized, alpha=alpha, clients=clients, run_id=run_id)
    console_path = paths.logs_dir / "run_console.log"
    console_path.write_text("", encoding="utf-8")
    _append_log(console_path, f"P7 HeteroFL starting | task={normalized} alpha={alpha} K={clients} mode={mode}")
    device = select_device(str(config["training"]["device"]))
    dropout = 0.2
    global_model = build_supernet(output_dim=spec.output_dim, dropout=dropout).to(device)
    client_data = {
        client_id: load_client_data(
            config,
            repo_root,
            scenario,
            spec,
            client_id=client_id,
            tier=mapping[client_id],
            max_samples=max_samples_per_client if mode == "smoke" else None,
        )
        for client_id in scenario.client_ids
    }
    validation = load_validation_union(
        config,
        repo_root,
        scenario,
        spec,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
    )
    bandwidth_static = round_bandwidth_by_tier(tier_mapping=mapping, output_dim=spec.output_dim, dropout=dropout)
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    slices_rows: list[dict[str, Any]] = []
    best_macro = -1.0
    best_round = 0
    best_state = None
    cumulative = 0
    for round_number in range(1, int(rounds) + 1):
        start = datetime.now()
        updates = []
        train_losses = []
        train_weights = []
        for client_id in scenario.client_ids:
            tier = mapping[client_id]
            data = client_data[client_id]
            sub_model = build_tier_model(tier=tier, output_dim=spec.output_dim, dropout=dropout).to(device)
            sub_state = extract_tier_state(global_model.state_dict(), tier)
            load_tier_state(sub_model, sub_state)
            result = train_local(
                model=sub_model,
                arrays=data.train,
                task=normalized,
                batch_size=int(config["training"]["batch_size"]),
                local_epochs=int(config["training"]["local_epochs"]),
                learning_rate=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
                device=device,
                seed=int(config["training"]["seed"]) + round_number,
            )
            val_local = evaluate_model(
                model=sub_model,
                arrays=data.val,
                task=normalized,
                class_names=spec.class_names,
                batch_size=int(config["training"]["batch_size"]) * 4,
                device=device,
                seed=int(config["training"]["seed"]),
            )
            size = model_size_bytes(sub_model)
            row = {
                "round": round_number,
                "client_id": client_id,
                "tier": tier,
                "train_samples": data.train.num_samples,
                "val_samples": data.val.num_samples,
                "local_loss": float(result["loss"]),
                "local_macro_f1": float(val_local["metrics"].get("macro_f1", 0.0)),
                "local_accuracy": float(val_local["metrics"].get("accuracy", 0.0)),
                "upload_bytes": size,
                "download_bytes": size,
                "model_size_bytes": size,
                "fit_time_sec": float(result["fit_time_sec"]),
            }
            client_rows.append(row)
            train_losses.append(float(result["loss"]))
            train_weights.append(int(data.train.num_samples))
            updates.append({"state_dict": sub_model.state_dict(), "num_examples": data.train.num_samples, "tier": tier})
        new_state, slice_info = aggregate_slice_weighted(updates, global_model.state_dict())
        global_model.load_state_dict(new_state, strict=True)
        val_global = evaluate_model(
            model=global_model,
            arrays=validation,
            task=normalized,
            class_names=spec.class_names,
            batch_size=int(config["training"]["batch_size"]) * 4,
            device=device,
            seed=int(config["training"]["seed"]),
        )
        cumulative += int(bandwidth_static["total_bytes"])
        elapsed = (datetime.now() - start).total_seconds()
        metrics = val_global["metrics"]
        row = {
            "round": round_number,
            "task": normalized,
            "alpha": float(alpha),
            "K": int(clients),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "attack_recall": float(metrics.get("recall_attack", 0.0)),
            "FPR": float(metrics.get("FPR", metrics.get("FPR_macro", 0.0))),
            "FNR": float(metrics.get("FNR", metrics.get("FNR_macro", 0.0))),
            "precision_macro": float(metrics.get("precision_macro", metrics.get("precision", 0.0))),
            "recall_macro": float(metrics.get("recall_macro", metrics.get("recall", 0.0))),
            "train_loss_mean": float(np.average(train_losses, weights=train_weights)) if train_weights else 0.0,
            "val_loss_mean": float(metrics.get("loss", 0.0)),
            "bandwidth_total_bytes": int(bandwidth_static["total_bytes"]),
            "bandwidth_cumulative_bytes": int(cumulative),
            "bandwidth_by_tier": json.dumps(bandwidth_static["by_tier"], sort_keys=True),
            "model_size_by_tier": json.dumps({k: v["model_size_bytes"] for k, v in bandwidth_static["by_client"].items()}, sort_keys=True),
            "slices_updated_ratio": float(slice_info["updated_ratio"]),
            "round_time_sec": float(elapsed),
        }
        round_rows.append(row)
        slices_rows.append({"round": round_number, **slice_info})
        _append_log(console_path, f"[Round {round_number:02d}/{rounds:02d}] macro_f1={row['macro_f1']:.4f} loss={row['train_loss_mean']:.4f} bytes={row['bandwidth_total_bytes']} slices={row['slices_updated_ratio']:.3f}")
        if row["macro_f1"] > best_macro:
            best_macro = row["macro_f1"]
            best_round = round_number
            best_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "round": best_round, "task": normalized}, paths.checkpoints_dir / "best_global_supernet.pth")
        torch.save({"model_state_dict": global_model.state_dict(), "round": round_number, "task": normalized}, paths.checkpoints_dir / "last_global_supernet.pth")
    if best_state is not None:
        global_model.load_state_dict(best_state, strict=True)
    test = load_global_test(config, repo_root, spec, max_samples=max_samples_per_client if mode == "smoke" else None)
    val_metrics = evaluate_model(model=global_model, arrays=validation, task=normalized, class_names=spec.class_names, batch_size=int(config["training"]["batch_size"]) * 4, device=device, seed=int(config["training"]["seed"]))["metrics"]
    test_metrics = evaluate_model(model=global_model, arrays=test, task=normalized, class_names=spec.class_names, batch_size=int(config["training"]["batch_size"]) * 4, device=device, seed=int(config["training"]["seed"]) + 99_000)["metrics"]
    comparison = _comparison(config, repo_root, normalized, alpha, clients, test_metrics)
    tier_summary = tier_parameter_summary(spec.output_dim)
    tier_rows = []
    for tier in ["weak", "medium", "powerful"]:
        tier_clients = [client for client, assigned in mapping.items() if assigned == tier]
        tier_client_rows = [row for row in client_rows if row["tier"] == tier]
        tier_rows.append({
            "tier": tier,
            "num_clients": len(tier_clients),
            "model_architecture": architecture_for_tier(tier, spec.output_dim),
            "num_parameters": tier_summary[tier]["num_parameters"],
            "model_size_bytes": tier_summary[tier]["model_size_bytes"],
            "avg_latency_ms_per_sample": 0.0,
            "avg_macro_f1": float(np.mean([row["local_macro_f1"] for row in tier_client_rows])) if tier_client_rows else 0.0,
            "bandwidth_total_bytes": int(sum(bandwidth_static["by_client"][client]["total_bytes"] for client in tier_clients) * int(rounds)),
        })
    _write_csv(paths.artifacts_dir / "metrics_rounds.csv", round_rows)
    _write_csv(paths.artifacts_dir / "metrics_clients.csv", client_rows)
    _write_csv(paths.artifacts_dir / "metrics_tiers.csv", tier_rows)
    _write_csv(paths.artifacts_dir / "bandwidth_by_tier.csv", tier_rows)
    _write_csv(paths.artifacts_dir / "slices_updated.csv", slices_rows)
    _write_csv(paths.artifacts_dir / "confusion_matrix.csv", _confusion_rows(test_metrics["confusion_matrix"], spec.class_names))
    from multitier_heterofl.config import write_json
    write_json(paths.artifacts_dir / "tier_mapping.json", mapping)
    write_json(paths.artifacts_dir / "tier_model_configs.json", tier_summary)
    write_json(paths.artifacts_dir / "model_config.json", {"supernet": tier_summary["supernet"], "task": normalized, "output_dim": spec.output_dim})
    write_json(paths.artifacts_dir / "metrics_val.json", val_metrics)
    write_json(paths.artifacts_dir / "metrics_test.json", test_metrics)
    write_json(paths.artifacts_dir / "classification_report.json", test_metrics.get("classification_report", {}))
    write_json(paths.artifacts_dir / "comparison_with_p4_p5_p6.json", comparison)
    figures_dir = repo_path(repo_root, config["outputs"]["figures_dir"]) / normalized / alpha_dir(alpha) / f"k{clients}" / paths.run_id
    generate_figures(figures_dir=figures_dir, task=normalized, round_rows=round_rows, tier_rows=tier_rows, metrics_test=test_metrics, comparison=comparison)
    manifest = {
        "phase": "P7",
        "run_id": paths.run_id,
        "task": normalized,
        "global_test_holdout": rel(scenario.global_test_npz, repo_root),
        "test_sent_to_clients": False,
        "run_dir": rel(paths.run_dir, repo_root),
        "figures_dir": rel(figures_dir, repo_root),
    }
    write_json(paths.artifacts_dir / "run_manifest.json", manifest)
    write_json(paths.run_dir / "manifest.json", manifest)
    write_json(paths.artifacts_dir / "run_summary.json", {"pending": True, "run_id": paths.run_id})
    artifacts = existing_relative(run_artifact_paths(paths.run_dir), repo_root)
    figures = existing_relative(run_figure_paths(figures_dir), repo_root)
    crit = criteria(artifacts=artifacts, figures=figures, task=normalized)
    summary = {
        "accepted": all(crit.values()),
        "phase": "P7",
        "task": normalized,
        "method": "HeteroFL",
        "implementation": "shared_supernet_prefix_slicing",
        "mode": mode,
        "run_id": paths.run_id,
        "scenario": {"alpha": float(alpha), "clients": int(clients), "rounds": int(rounds)},
        "tiers": mapping,
        "models": {
            "weak": architecture_for_tier("weak"),
            "medium": architecture_for_tier("medium"),
            "powerful": architecture_for_tier("powerful"),
            "supernet": architecture_for_tier("supernet"),
        },
        "dataset": {
            "train_rows_total": int(sum(data.train.num_samples for data in client_data.values())),
            "val_rows_total": int(sum(data.val.num_samples for data in client_data.values())),
            "test_rows": int(test_metrics.get("support_total", 0)),
            "client_train_rows": {cid: data.train.num_samples for cid, data in client_data.items()},
            "client_val_rows": {cid: data.val.num_samples for cid, data in client_data.items()},
            "global_test_holdout": rel(scenario.global_test_npz, repo_root),
            "test_sent_to_clients": False,
        },
        "training": {"rounds_completed": int(rounds), "best_round": int(best_round), "local_epochs": int(config["training"]["local_epochs"])},
        "validation": {"metrics": val_metrics},
        "test": {"metrics": test_metrics},
        "communication": {"total_bytes": int(cumulative), "by_tier": {row["tier"]: row["bandwidth_total_bytes"] for row in tier_rows}},
        "comparison": comparison,
        "artifacts": artifacts,
        "figures": figures,
        "criteria": crit,
        "warnings": ["Smoke run uses sampled clients/test data; metrics are not scientifically significant."] if mode == "smoke" else [],
        "errors": [],
    }
    write_json(paths.artifacts_dir / "run_summary.json", summary)
    write_json(paths.scenario_dir / "latest_run_summary.json", summary)
    _append_log(console_path, f"P7 HeteroFL finished | accepted={summary['accepted']} best_round={best_round}")
    return summary
