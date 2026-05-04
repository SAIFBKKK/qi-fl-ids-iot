from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.common.paths import ARTIFACTS_DIR, DATA_DIR, OUTPUTS_DIR
from src.data.dataset import IoTLocalDataset
from src.model.network import MLPClassifier


REPORT_DIR = OUTPUTS_DIR / "reports" / "qi_benchmark_reduced"
BASELINE_DIR = OUTPUTS_DIR / "reports" / "baselines"
CONFUSION_DIR = REPORT_DIR / "confusion_matrices"
RARE_CLASS_IDS = (0, 3, 30, 31, 33)

TARGETS = [
    ("E1", "exp_bench30_normal_fedavg_28f", "normal_noniid"),
    ("E2", "exp_bench30_normal_qifa_28f", "normal_noniid"),
    ("E5", "exp_bench30_absent_fedavg_28f", "absent_local"),
    ("E6", "exp_bench30_absent_qifa_28f", "absent_local"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_indices(config: dict[str, Any], scenario: str) -> list[int] | None:
    model_cfg = dict(config.get("model", {}))
    feature_cfg = dict(config.get("feature_selection", {}))
    feature_cfg.update(dict(model_cfg.get("feature_selection", {})))
    if not bool(feature_cfg.get("enabled", False)):
        return None
    raw_path = str(feature_cfg.get("artifact_path", "artifacts/qi_feature_selection/{scenario}/selected_features.json")).format(scenario=scenario)
    path = Path(raw_path)
    if not path.is_absolute():
        path = ARTIFACTS_DIR.parent / path
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    return [int(idx) for idx in payload["selected_indices"]]


def _build_model(config: dict[str, Any], input_dim: int) -> MLPClassifier:
    model_cfg = dict(config.get("model", {}))
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=int(model_cfg.get("output_dim", config.get("dataset", {}).get("num_classes", 34))),
        hidden_dims=tuple(model_cfg.get("hidden_dims", [256, 128])),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def _load_checkpoint_model(experiment_name: str, scenario: str) -> tuple[MLPClassifier, dict[str, Any], list[int] | None]:
    run_dir = BASELINE_DIR / experiment_name
    resolved = _load_json(run_dir / "resolved_config.json")
    if resolved is None:
        raise FileNotFoundError(run_dir / "resolved_config.json")
    config = dict(resolved["config"])
    indices = _selected_indices(config, scenario)
    input_dim = len(indices) if indices is not None else int(config.get("model", {}).get("input_dim", 28))
    model = _build_model(config, input_dim)
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config, indices


def _load_eval_data(scenario: str, selected_indices: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for node_id in ("node1", "node2", "node3"):
        path = DATA_DIR / "processed" / scenario / node_id / "val_preprocessed.npz"
        dataset = IoTLocalDataset(path, selected_feature_indices=selected_indices)
        X_parts.append(dataset.X.numpy())
        y_parts.append(dataset.y.numpy())
    return np.vstack(X_parts), np.concatenate(y_parts)


def _predict(model: MLPClassifier, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32)
            preds.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def _save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def _plot_matrix(matrix: np.ndarray, title: str, output_path: Path, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ticks = np.arange(0, matrix.shape[0], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    for class_id in RARE_CLASS_IDS:
        ax.axhline(class_id - 0.5, color="red", alpha=0.25, linewidth=0.8)
        ax.axvline(class_id - 0.5, color="red", alpha=0.25, linewidth=0.8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    suffix = "normalized" if normalized else "raw"
    ax.text(
        0.01,
        -0.12,
        f"Red guide lines mark rare classes: {RARE_CLASS_IDS}. Matrix type: {suffix}.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom != 0)


def evaluate_target(experiment_id: str, experiment_name: str, scenario: str) -> dict[str, Any]:
    out_dir = CONFUSION_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model, config, indices = _load_checkpoint_model(experiment_name, scenario)
    X, y_true = _load_eval_data(scenario, indices)
    y_pred = _predict(model, X)
    labels = list(range(int(config.get("dataset", {}).get("num_classes", 34))))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = _normalize_rows(matrix)

    _save_matrix_csv(out_dir / "confusion_matrix_raw.csv", matrix)
    _save_matrix_csv(out_dir / "confusion_matrix_normalized.csv", normalized)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    support = matrix.sum(axis=1)
    dominant_classes = [int(idx) for idx in np.argsort(support)[::-1][:5]]
    (out_dir / "class_notes.json").write_text(
        json.dumps(
            {
                "rare_class_ids": list(RARE_CLASS_IDS),
                "dominant_class_ids": dominant_classes,
                "scenario": scenario,
                "experiment_name": experiment_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_matrix(matrix, f"{experiment_id} raw confusion matrix", out_dir / "confusion_matrix_raw.png", normalized=False)
    _plot_matrix(normalized, f"{experiment_id} normalized confusion matrix", out_dir / "confusion_matrix_normalized.png", normalized=True)
    return {"experiment_id": experiment_id, "matrix": matrix, "normalized": normalized}


def build_combined_figure(results: list[dict[str, Any]]) -> None:
    figure_dir = REPORT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Confusion matrices pending: E1/E2/E5/E6 30-round checkpoints are not available yet.",
            ha="center",
            va="center",
            wrap=True,
        )
        fig.tight_layout()
        fig.savefig(figure_dir / "figure4_confusion_matrices.png", dpi=170)
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 5 * len(results)))
    if len(results) == 1:
        axes = np.asarray([axes])
    for row_idx, result in enumerate(results):
        for col_idx, key in enumerate(("matrix", "normalized")):
            ax = axes[row_idx, col_idx]
            matrix = result[key]
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_title(f"{result['experiment_id']} {'normalized' if key == 'normalized' else 'raw'}")
            ticks = np.arange(0, matrix.shape[0], 8)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            for class_id in RARE_CLASS_IDS:
                ax.axhline(class_id - 0.5, color="red", alpha=0.25, linewidth=0.8)
                ax.axvline(class_id - 0.5, color="red", alpha=0.25, linewidth=0.8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure4_confusion_matrices.png", dpi=170)
    plt.close(fig)


def main() -> None:
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for target in TARGETS:
        try:
            results.append(evaluate_target(*target))
        except Exception as exc:
            failures.append({"experiment_id": target[0], "experiment_name": target[1], "error": str(exc)})
    build_combined_figure(results)
    (CONFUSION_DIR / "confusion_status.json").write_text(
        json.dumps({"completed": [r["experiment_id"] for r in results], "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(f"Confusion matrix outputs -> {CONFUSION_DIR}")


if __name__ == "__main__":
    main()
