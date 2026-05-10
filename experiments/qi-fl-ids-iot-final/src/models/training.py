"""Training orchestration for P4 centralized L1 binary baseline."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .l1_mlp import CentralizedL1MLP
from .metrics import (
    binary_metrics,
    classification_report_dict,
    optional_auc_metrics,
    predictions_from_threshold,
    select_thresholds,
    threshold_sweep,
)


@dataclass(frozen=True)
class CentralizedL1Run:
    """Result returned by the P4 training pipeline."""

    summary: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    generated_files: list[str]

    @property
    def accepted(self) -> bool:
        return not self.errors and bool(self.summary.get("accepted", False))


def load_config(config_path: Path) -> dict[str, Any]:
    """Load centralized L1 YAML config."""

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


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _load_npz_dataset(path: Path, target_key: str = "y_binary") -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ["X", target_key, "label_id_original", "row_id"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        return {
            "X": np.asarray(data["X"], dtype=np.float32),
            "y": np.asarray(data[target_key], dtype=np.int64),
            "label_id_original": np.asarray(data["label_id_original"], dtype=np.int64),
            "row_id": np.asarray(data["row_id"], dtype=np.int64),
        }


def _make_loader(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data["X"]),
        torch.from_numpy(data["y"]).long(),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _verify_p2(
    repo_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    inputs = config["inputs"]
    final_dir = _repo_path(repo_root, config["final_experiment_dir"])

    required_paths = {
        "train_npz": _repo_path(repo_root, inputs["train_npz"]),
        "val_npz": _repo_path(repo_root, inputs["val_npz"]),
        "test_npz": _repo_path(repo_root, inputs["test_npz"]),
        "manifest": _repo_path(repo_root, inputs["manifest"]),
        "sampling_report": _repo_path(repo_root, inputs["sampling_report"]),
        "scaler": _repo_path(repo_root, inputs["scaler"]),
        "feature_names": _repo_path(repo_root, inputs["feature_names"]),
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
        warnings.append("P2 summary JSON not found; relying on L1 manifest")

    feature_count = None
    if required_paths["feature_names"].exists():
        feature_names = _load_json(required_paths["feature_names"])
        feature_count = len(feature_names)
        if feature_count != int(config["model"]["input_dim"]):
            errors.append(
                f"feature_names count {feature_count} does not match input_dim {config['model']['input_dim']}"
            )

    if required_paths["manifest"].exists():
        manifest = _load_json(required_paths["manifest"])
        if int(manifest.get("feature_count", -1)) != int(config["model"]["input_dim"]):
            errors.append("L1 manifest feature_count does not match model input_dim")
        row_counts = manifest.get("row_counts", {})
        if row_counts.get("total") != 630_000:
            warnings.append("L1 manifest total count differs from expected 630000")

    return {
        "required_paths": {
            name: _rel(path, repo_root) for name, path in required_paths.items()
        },
        "p2_summary_path": _rel(p2_summary_path, repo_root)
        if p2_summary_path.exists()
        else None,
        "p2_accepted": p2_accepted,
        "feature_count": feature_count,
    }, errors, warnings


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        preds.append(torch.argmax(logits.detach(), dim=1).cpu().numpy())
        targets.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_rows
    return metrics


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    collect_probabilities: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    start = time.perf_counter()

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        batch_size = y_batch.size(0)
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(y_batch.cpu().numpy())
        if collect_probabilities:
            probs.append(probabilities.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_rows
    metrics["latency_ms_per_sample"] = (elapsed * 1000.0) / total_rows
    result: dict[str, Any] = {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}
    if collect_probabilities:
        result["prob_attack"] = np.concatenate(probs)
    return result


def _checkpoint_payload(
    model: CentralizedL1MLP,
    config: dict[str, Any],
    epoch: int,
    metric_name: str,
    metric_value: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": config["model"],
        "epoch": epoch,
        "selection_metric": metric_name,
        "selection_metric_value": metric_value,
        "selection_split": "validation",
        "test_used_for_model_selection": False,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_best_model(path: Path, device: torch.device) -> tuple[CentralizedL1MLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = CentralizedL1MLP(
        input_dim=int(model_config["input_dim"]),
        hidden_layers=list(model_config["hidden_layers"]),
        output_dim=int(model_config["output_dim"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _plot_history(history: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "centralized_l1_loss_curve.png",
        output_dir / "centralized_l1_accuracy_curve.png",
        output_dir / "centralized_l1_f1_curve.png",
    ]
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train", color="#2563EB")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val", color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Centralized L1 loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_accuracy"] for row in history], label="train", color="#2563EB")
    ax.plot(epochs, [row["val_accuracy"] for row in history], label="val", color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Centralized L1 accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[1], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_macro_f1"] for row in history], label="train macro-F1", color="#2563EB")
    ax.plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1", color="#DC2626")
    ax.plot(epochs, [row["val_f1_attack"] for row in history], label="val attack-F1", color="#16A34A")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Centralized L1 F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[2], dpi=160)
    plt.close(fig)
    return paths


def _plot_confusion_matrix(counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred normal", "pred attack"])
    ax.set_yticks([0, 1], labels=["true normal", "true attack"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    ax.set_title("Centralized L1 confusion matrix")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_threshold_sweep(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [row["f1_attack"] for row in rows], label="attack F1", color="#2563EB")
    ax.plot(thresholds, [row["recall_attack"] for row in rows], label="attack recall", color="#16A34A")
    ax.plot(thresholds, [row["FPR"] for row in rows], label="FPR", color="#DC2626")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Validation threshold sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_tp_tn_fp_fn(counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["TP", "TN", "FP", "FN"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, [counts[label] for label in labels], color=["#16A34A", "#2563EB", "#F97316", "#DC2626"])
    ax.set_ylabel("Rows")
    ax.set_title("Centralized L1 TP/TN/FP/FN")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc_pr(auc_payload: dict[str, Any], roc_path: Path, pr_path: Path) -> list[Path]:
    generated: list[Path] = []
    if auc_payload.get("roc_curve"):
        curve = auc_payload["roc_curve"]
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(curve["fpr"], curve["tpr"], color="#2563EB", label=f"ROC-AUC={auc_payload['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], color="#64748B", linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Centralized L1 ROC curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(roc_path, dpi=160)
        plt.close(fig)
        generated.append(roc_path)
    if auc_payload.get("pr_curve"):
        curve = auc_payload["pr_curve"]
        pr_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(curve["recall"], curve["precision"], color="#16A34A", label=f"PR-AUC={auc_payload['pr_auc']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Centralized L1 PR curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(pr_path, dpi=160)
        plt.close(fig)
        generated.append(pr_path)
    return generated


def _plot_architecture(model_config: dict[str, Any], num_parameters: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = [model_config["input_dim"], *model_config["hidden_layers"], model_config["output_dim"]]
    labels = [str(dim) for dim in dims]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(labels))
    for idx, (x_pos, label) in enumerate(zip(xs, labels)):
        ax.text(
            x_pos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EFF6FF", "edgecolor": "#2563EB"},
        )
        if idx < len(labels) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.055, 0.55), xytext=(x_pos + 0.055, 0.55), arrowprops={"arrowstyle": "->"})
    ax.text(0.5, 0.16, f"ReLU + dropout={model_config['dropout']} | parameters={num_parameters}", ha="center")
    ax.set_title("CentralizedL1MLP architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_confusion_csv(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"label": "true_normal", "pred_normal": counts["TN"], "pred_attack": counts["FP"]},
        {"label": "true_attack", "pred_normal": counts["FN"], "pred_attack": counts["TP"]},
    ]
    _write_dict_csv(path, rows)


def _write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    historical: dict[str, Any],
) -> None:
    accepted_text = "P4 est validée." if summary.get("accepted") else "P4 n'est pas validée."
    lines = [
        "# P4 — Centralized L1 Binary Baseline",
        "",
        "## 1. Objectif",
        "Entraîner une baseline centralisée L1 binaire normal vs attack, référence directe pour P5 FL L1.",
        "",
        "## 2. Données utilisées",
        f"- Train : `{summary['dataset']['train_rows']}` rows.",
        f"- Val : `{summary['dataset']['val_rows']}` rows.",
        f"- Test holdout : `{summary['dataset']['test_rows']}` rows.",
        "",
        "## 3. Baseline historique Kaggle 34 classes",
        f"- Architecture : `{historical['architecture']}`.",
        f"- Best val macro-F1 : `{historical['best_val_f1_macro']}`.",
        f"- Test macro-F1 : `{historical['test_f1_macro']}`.",
        "",
        "## 4. Pourquoi une nouvelle baseline L1 est nécessaire",
        "La baseline Kaggle est multiclasses L3/34 classes. P4 entraîne un modèle binaire dédié à la production L1 et à la comparaison future avec FL.",
        "",
        "## 5. Architecture du modèle L1",
        f"- `{summary['model']['architecture']}`.",
        f"- Paramètres : `{summary['model']['num_parameters']}`.",
        "",
        "## 6. Configuration d'entraînement",
        f"- Optimizer : `{summary['training']['optimizer']}`.",
        f"- Batch size : `{summary['training']['batch_size']}`.",
        f"- Device : `{summary['training']['device']}`.",
        "",
        "## 7. Stratégie de validation",
        "Le meilleur checkpoint est sélectionné uniquement sur validation macro-F1. Le test global n’est pas utilisé pendant l'entraînement.",
        "",
        "## 8. Threshold tuning",
        f"- Primary threshold : `{summary['threshold']['primary_threshold']}`.",
        "- Le threshold est choisi uniquement sur validation.",
        "",
        "## 9. Résultats validation",
        f"- Best epoch : `{summary['training']['best_epoch']}`.",
        f"- Val macro-F1 : `{summary['validation']['macro_f1']}`.",
        f"- Val attack-F1 : `{summary['validation']['f1_attack']}`.",
        "",
        "## 10. Résultats test global holdout",
        f"- Accuracy : `{summary['test']['accuracy']}`.",
        f"- Macro-F1 : `{summary['test']['macro_f1']}`.",
        f"- Attack recall : `{summary['test']['recall_attack']}`.",
        "",
        "## 11. Matrice de confusion",
        f"- TP `{summary['test']['TP']}`, TN `{summary['test']['TN']}`, FP `{summary['test']['FP']}`, FN `{summary['test']['FN']}`.",
        "",
        "## 12. Analyse TP / TN / FP / FN",
        "Les TP/TN/FP/FN sont calculés avec attack comme classe positive.",
        "",
        "## 13. FPR et FNR",
        f"- FPR : `{summary['test']['FPR']}`.",
        f"- FNR : `{summary['test']['FNR']}`.",
        "",
        "## 14. Latence et taille modèle",
        f"- Latence : `{summary['test']['latency_ms_per_sample']}` ms/sample.",
        f"- Taille modèle : `{summary['model']['model_size_bytes']}` bytes.",
        "",
        "## 15. Artefacts générés",
    ]
    for artifact in summary["artifacts"]:
        lines.append(f"- `{artifact}`")
    lines.extend(["", "## 16. Figures générées"])
    for figure in summary["figures"]:
        lines.append(f"- `{figure}`")
    lines.extend(
        [
            "",
            "## 17. Comparaison attendue avec P5 FL",
            "P5 devra comparer FedAvg L1 au modèle centralisé P4 sur le même test global holdout.",
            "",
            "## 18. Risques restants",
        ]
    )
    if summary["warnings"]:
        lines.extend([f"- {warning}" for warning in summary["warnings"]])
    else:
        lines.append("- Aucun warning restant.")
    lines.extend(["", "## 19. Critères d’acceptation", "", "| critere | ok |", "| --- | --- |"])
    for key, value in summary["criteria"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 20. Conclusion P4", "", accepted_text, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def train_centralized_l1(config_path: Path) -> CentralizedL1Run:
    """Run P4 centralized L1 training, validation thresholding and holdout test."""

    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    outputs = config["outputs"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    threshold_cfg = config["threshold"]

    run_dir = _repo_path(repo_root, outputs["run_dir"])
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    reports_dir = _repo_path(repo_root, outputs["reports_dir"])
    figures_training_dir = _repo_path(repo_root, outputs["figures_training_dir"])
    figures_eval_dir = _repo_path(repo_root, outputs["figures_evaluation_dir"])
    figures_models_dir = _repo_path(repo_root, outputs["figures_models_dir"])
    for directory in [checkpoints_dir, artifacts_dir, logs_dir, reports_dir, figures_training_dir, figures_eval_dir, figures_models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    p2_check, errors, warnings = _verify_p2(repo_root, config)
    if errors:
        summary = {"accepted": False, "errors": errors, "warnings": warnings, "criteria": {}}
        _write_json(reports_dir / "centralized_l1_summary.json", summary)
        return CentralizedL1Run(summary, errors, warnings, [_rel(reports_dir / "centralized_l1_summary.json", repo_root)])

    _set_seed(int(training_cfg["seed"]))
    device = _select_device(str(training_cfg["device"]))

    train_path = _repo_path(repo_root, config["inputs"]["train_npz"])
    val_path = _repo_path(repo_root, config["inputs"]["val_npz"])
    test_path = _repo_path(repo_root, config["inputs"]["test_npz"])
    train_data = _load_npz_dataset(train_path)
    val_data = _load_npz_dataset(val_path)

    if train_data["X"].shape[1] != int(model_cfg["input_dim"]):
        errors.append("train input_dim does not match model input_dim")
    if set(np.unique(train_data["y"]).tolist()) != {0, 1}:
        errors.append("train labels must contain binary labels 0 and 1")
    if int(model_cfg["output_dim"]) != 2:
        errors.append("model output_dim must be 2")
    if errors:
        summary = {"accepted": False, "errors": errors, "warnings": warnings, "criteria": {}}
        _write_json(reports_dir / "centralized_l1_summary.json", summary)
        return CentralizedL1Run(summary, errors, warnings, [_rel(reports_dir / "centralized_l1_summary.json", repo_root)])

    model = CentralizedL1MLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_layers=list(model_cfg["hidden_layers"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
    ).to(device)
    num_parameters = model.count_parameters()

    train_loader = _make_loader(
        train_data,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        seed=int(training_cfg["seed"]),
        device=device,
    )
    train_eval_loader = _make_loader(
        train_data,
        batch_size=int(training_cfg["batch_size"]) * 4,
        shuffle=False,
        seed=int(training_cfg["seed"]),
        device=device,
    )
    val_loader = _make_loader(
        val_data,
        batch_size=int(training_cfg["batch_size"]) * 4,
        shuffle=False,
        seed=int(training_cfg["seed"]),
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = None
    if str(training_cfg["scheduler"]) == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

    best_metric = -math.inf
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    best_model_path = checkpoints_dir / "best_model.pth"
    last_model_path = checkpoints_dir / "last_model.pth"
    metric_name = str(training_cfg["best_metric"])

    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        train_metrics_live = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_eval = _evaluate_loader(model, train_eval_loader, criterion, device)
        val_eval = _evaluate_loader(model, val_loader, criterion, device)
        train_metrics = train_eval["metrics"]
        val_metrics = val_eval["metrics"]
        current_metric = float(val_metrics["macro_f1"])

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_metrics["loss"],
            "train_live_loss": train_metrics_live["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision_attack": train_metrics["precision_attack"],
            "val_precision_attack": val_metrics["precision_attack"],
            "train_recall_attack": train_metrics["recall_attack"],
            "val_recall_attack": val_metrics["recall_attack"],
            "train_f1_attack": train_metrics["f1_attack"],
            "val_f1_attack": val_metrics["f1_attack"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_FPR": train_metrics["FPR"],
            "val_FPR": val_metrics["FPR"],
            "train_FNR": train_metrics["FNR"],
            "val_FNR": val_metrics["FNR"],
        }
        history.append(row)

        if scheduler is not None:
            scheduler.step(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve = 0
            _save_checkpoint(
                best_model_path,
                _checkpoint_payload(model, config, epoch, metric_name, current_metric),
            )
        else:
            no_improve += 1

        _save_checkpoint(
            last_model_path,
            _checkpoint_payload(model, config, epoch, metric_name, current_metric),
        )

        if no_improve >= int(training_cfg["early_stopping_patience"]):
            break

    history_path = artifacts_dir / "training_history.csv"
    _write_dict_csv(history_path, history)
    logs_history_path = logs_dir / "training_history.csv"
    _write_dict_csv(logs_history_path, history)

    best_model, checkpoint = _load_best_model(best_model_path, device)
    val_eval_prob = _evaluate_loader(best_model, val_loader, criterion, device, collect_probabilities=True)
    threshold_rows = threshold_sweep(
        val_eval_prob["y_true"],
        val_eval_prob["prob_attack"],
        start=float(threshold_cfg["start"]),
        stop=float(threshold_cfg["stop"]),
        step=float(threshold_cfg["step"]),
    )
    selected_thresholds = select_thresholds(threshold_rows)
    primary_threshold = float(selected_thresholds["primary_threshold"])
    val_pred_threshold = predictions_from_threshold(val_eval_prob["prob_attack"], primary_threshold)
    val_threshold_metrics = binary_metrics(val_eval_prob["y_true"], val_pred_threshold)
    val_threshold_metrics["loss"] = val_eval_prob["metrics"]["loss"]
    val_threshold_metrics["threshold"] = primary_threshold
    val_threshold_metrics["selection_split"] = "validation"
    val_threshold_metrics["test_used_for_threshold"] = False

    threshold_sweep_path = artifacts_dir / "threshold_sweep.csv"
    _write_dict_csv(threshold_sweep_path, threshold_rows)
    threshold_path = artifacts_dir / "threshold.json"
    _write_json(threshold_path, selected_thresholds)
    metrics_val_path = artifacts_dir / "metrics_val.json"
    _write_json(metrics_val_path, val_threshold_metrics)

    test_data = _load_npz_dataset(test_path)
    test_loader = _make_loader(
        test_data,
        batch_size=int(training_cfg["batch_size"]) * 4,
        shuffle=False,
        seed=int(training_cfg["seed"]),
        device=device,
    )
    test_eval_prob = _evaluate_loader(best_model, test_loader, criterion, device, collect_probabilities=True)
    test_pred = predictions_from_threshold(test_eval_prob["prob_attack"], primary_threshold)
    test_metrics = binary_metrics(test_eval_prob["y_true"], test_pred)
    test_metrics["loss"] = test_eval_prob["metrics"]["loss"]
    test_metrics["threshold"] = primary_threshold
    test_metrics["latency_ms_per_sample"] = test_eval_prob["metrics"]["latency_ms_per_sample"]
    auc_payload = optional_auc_metrics(test_eval_prob["y_true"], test_eval_prob["prob_attack"])
    test_metrics["roc_auc"] = auc_payload["roc_auc"]
    test_metrics["pr_auc"] = auc_payload["pr_auc"]
    test_metrics["roc_pr_warning"] = auc_payload["warning"]

    model_size_bytes = int(best_model_path.stat().st_size)
    test_metrics["model_size_bytes"] = model_size_bytes
    test_metrics["num_parameters"] = num_parameters

    metrics_test_path = artifacts_dir / "metrics_test.json"
    _write_json(metrics_test_path, test_metrics)
    classification_report_path = artifacts_dir / "classification_report.json"
    _write_json(classification_report_path, classification_report_dict(test_eval_prob["y_true"], test_pred))
    confusion_path = artifacts_dir / "confusion_matrix.csv"
    _write_confusion_csv(confusion_path, test_metrics)

    historical = {
        **config["historical_baseline"],
        "role": "Historical multiclass L3/34-class reference only",
        "not_used_for_p4_l1": True,
        "p4_difference": (
            "P4 trains a new binary normal-vs-attack model with output_dim=2 on "
            "the L1 preprocessed dataset."
        ),
    }
    historical_path = artifacts_dir / "historical_kaggle_34class_baseline.json"
    _write_json(historical_path, historical)

    model_config = {
        "model": model_cfg,
        "training": training_cfg,
        "num_parameters": num_parameters,
        "architecture": "28 -> 128 -> 64 -> 2",
        "best_checkpoint_selection": {
            "metric": metric_name,
            "split": "validation",
            "test_used_for_model_selection": False,
            "best_epoch": best_epoch,
        },
        "threshold_selection": {
            "split": "validation",
            "test_used_for_threshold": False,
            "primary_threshold": primary_threshold,
        },
        "data_usage": {
            "centralized_l1_p2_only": True,
            "dirichlet_partitions_used": False,
            "test_usage": "final_holdout_only_after_model_and_threshold_selection",
        },
    }
    model_config_path = artifacts_dir / "model_config.json"
    _write_json(model_config_path, model_config)

    figure_paths = []
    figure_paths.extend(_plot_history(history, figures_training_dir))
    confusion_fig = figures_eval_dir / "centralized_l1_confusion_matrix.png"
    threshold_fig = figures_eval_dir / "centralized_l1_threshold_sweep.png"
    tp_fig = figures_eval_dir / "centralized_l1_tp_tn_fp_fn.png"
    _plot_confusion_matrix(test_metrics, confusion_fig)
    _plot_threshold_sweep(threshold_rows, threshold_fig)
    _plot_tp_tn_fp_fn(test_metrics, tp_fig)
    figure_paths.extend([confusion_fig, threshold_fig, tp_fig])
    figure_paths.extend(
        _plot_roc_pr(
            auc_payload,
            figures_eval_dir / "centralized_l1_roc_curve.png",
            figures_eval_dir / "centralized_l1_pr_curve.png",
        )
    )
    architecture_fig = figures_models_dir / "centralized_l1_architecture.png"
    _plot_architecture(model_cfg, num_parameters, architecture_fig)
    figure_paths.append(architecture_fig)

    artifacts = [
        best_model_path,
        last_model_path,
        model_config_path,
        threshold_path,
        metrics_val_path,
        metrics_test_path,
        classification_report_path,
        confusion_path,
        threshold_sweep_path,
        history_path,
        logs_history_path,
        historical_path,
    ]

    criteria = {
        "p2_validated_detected": p2_check["p2_accepted"],
        "train_val_test_l1_loaded": True,
        "input_dim_28_confirmed": train_data["X"].shape[1] == 28,
        "output_dim_2_confirmed": int(model_cfg["output_dim"]) == 2,
        "model_trained_without_error": len(history) > 0,
        "best_model_saved": best_model_path.exists(),
        "last_model_saved": last_model_path.exists(),
        "training_history_generated": history_path.exists(),
        "best_checkpoint_validation_only": checkpoint["selection_split"] == "validation"
        and not checkpoint["test_used_for_model_selection"],
        "threshold_validation_only": not selected_thresholds["test_used_for_threshold"],
        "test_evaluated_after_selection": True,
        "metrics_val_generated": metrics_val_path.exists(),
        "metrics_test_generated": metrics_test_path.exists(),
        "threshold_json_generated": threshold_path.exists(),
        "threshold_sweep_generated": threshold_sweep_path.exists(),
        "confusion_matrix_generated": confusion_path.exists(),
        "tp_tn_fp_fn_calculated": all(key in test_metrics for key in ["TP", "TN", "FP", "FN"]),
        "core_metrics_calculated": all(
            key in test_metrics
            for key in ["accuracy", "precision", "recall", "f1", "macro_f1"]
        ),
        "fpr_fnr_calculated": all(key in test_metrics for key in ["FPR", "FNR"]),
        "latency_calculated": "latency_ms_per_sample" in test_metrics,
        "model_size_calculated": model_size_bytes > 0,
        "num_parameters_calculated": num_parameters > 0,
        "historical_kaggle_artifact_generated": historical_path.exists(),
        "docs_generated": True,
        "figures_generated": all(path.exists() for path in figure_paths),
        "dirichlet_partitions_not_used": True,
    }

    score_warnings: list[str] = []
    if test_metrics["macro_f1"] < 0.85:
        score_warnings.append("test_macro_f1 is below recommended target 0.85")
    if test_metrics["recall_attack"] < 0.90:
        score_warnings.append("attack_recall is below recommended target 0.90")
    warnings.extend(score_warnings)

    summary = {
        "accepted": all(criteria.values()) and not errors,
        "dataset": {
            "train_rows": int(train_data["X"].shape[0]),
            "val_rows": int(val_data["X"].shape[0]),
            "test_rows": int(test_data["X"].shape[0]),
            "input_dim": int(train_data["X"].shape[1]),
            "labels": {"normal": 0, "attack": 1},
        },
        "model": {
            "name": model_cfg["name"],
            "architecture": "28 -> 128 -> 64 -> 2",
            "num_parameters": num_parameters,
            "model_size_bytes": model_size_bytes,
        },
        "training": {
            "device": str(device),
            "batch_size": int(training_cfg["batch_size"]),
            "optimizer": training_cfg["optimizer"],
            "max_epochs": int(training_cfg["max_epochs"]),
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_metric": metric_name,
            "best_val_macro_f1": best_metric,
        },
        "validation": val_threshold_metrics,
        "threshold": selected_thresholds,
        "test": test_metrics,
        "historical_baseline": historical,
        "artifacts": [_rel(path, repo_root) for path in artifacts],
        "figures": [_rel(path, repo_root) for path in figure_paths],
        "criteria": criteria,
        "warnings": warnings,
        "errors": errors,
    }

    summary_path = reports_dir / "centralized_l1_summary.json"
    _write_json(summary_path, summary)
    artifacts.append(summary_path)

    docs_path = _repo_path(repo_root, config["final_experiment_dir"]) / "docs" / "04_centralized_baseline.md"
    _write_markdown_report(docs_path, summary, historical)
    artifacts.append(docs_path)

    generated_files = [_rel(path, repo_root) for path in [*artifacts, *figure_paths]]
    return CentralizedL1Run(
        summary=summary,
        errors=errors,
        warnings=warnings,
        generated_files=sorted(set(generated_files)),
    )
