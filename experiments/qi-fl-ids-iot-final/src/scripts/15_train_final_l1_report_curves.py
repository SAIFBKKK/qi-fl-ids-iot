"""Train final L1 report model and generate publication-ready curves.

This script is intentionally separate from the FL evidence pipeline. It trains a
local L1 binary MLP for report figures only, using train/validation splits for
optimization and early stopping, then evaluates the held-out test split once at
the end.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


REPO_DEFAULT_FEATURE_NAMES = Path(
    "experiments/qi-fl-ids-iot-final/outputs/artifacts/features/feature_names.json"
)


def load_feature_names_json(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("feature_names", payload.get("features"))

    if not isinstance(payload, list):
        raise ValueError(f"Unsupported feature names format in {path}")

    return [str(item) for item in payload]


def _extract_labels(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    for key in ("y", "y_binary", "labels", "label", "target"):
        if key in data:
            return data[key].astype(np.int64)
    raise KeyError(f"No label array found in {path}; expected one of y/y_binary/labels/label/target")


def load_npz_dataset(path: Path, feature_names_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"Missing X array in {path}")

    X = data["X"].astype(np.float32)
    y = _extract_labels(data, path)

    feature_names: list[str] | None = None
    if "feature_names" in data:
        raw_names = data["feature_names"]
        feature_names = [str(item) for item in raw_names.tolist()]
    else:
        feature_names = load_feature_names_json(feature_names_path)

    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Feature names length mismatch for {path}: {len(feature_names)} names for {X.shape[1]} columns"
        )

    return X, y, feature_names


def load_qga_selected_features(qga_dir: Path) -> tuple[list[str], list[int] | None, dict[str, Any]]:
    selected_features_path = qga_dir / "selected_features.json"
    decision_path = qga_dir / "selection_decision.json"

    if not selected_features_path.exists():
        raise FileNotFoundError(f"Missing selected_features.json: {selected_features_path}")
    if not decision_path.exists():
        raise FileNotFoundError(f"Missing selection_decision.json: {decision_path}")

    selected_payload = json.loads(selected_features_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    if isinstance(selected_payload, dict):
        selected_features = selected_payload.get("selected_features")
        selected_indices = selected_payload.get("selected_indices")
        selected_mask_id = decision.get("selected_mask_id") or selected_payload.get("mask_id")
        features_count = (
            decision.get("features_count")
            or decision.get("selected_features_count")
            or selected_payload.get("selected_features_count")
            or (len(selected_features) if isinstance(selected_features, list) else None)
        )
    else:
        selected_features = selected_payload
        selected_indices = None
        selected_mask_id = decision.get("selected_mask_id")
        features_count = decision.get("features_count") or decision.get("selected_features_count")

    if not isinstance(selected_features, list):
        raise ValueError(f"Unsupported selected features format in {selected_features_path}")

    selected_features = [str(item) for item in selected_features]
    selected_indices = [int(idx) for idx in selected_indices] if isinstance(selected_indices, list) else None
    features_count = int(features_count if features_count is not None else len(selected_features))

    if selected_mask_id != "conservative_seed_42":
        raise ValueError(f"Expected selected_mask_id='conservative_seed_42', got {selected_mask_id!r}")
    if features_count != 12:
        raise ValueError(f"Expected 12 selected features for final QGA deployment model, got {features_count}")
    if len(selected_features) != 12:
        raise ValueError(f"selected_features.json should contain 12 features, got {len(selected_features)}")

    merged_decision = dict(decision)
    merged_decision["selected_mask_id"] = selected_mask_id
    merged_decision["features_count"] = features_count
    return selected_features, selected_indices, merged_decision


def apply_feature_selection(
    X: np.ndarray,
    feature_names: list[str] | None,
    selected_features: list[str],
    selected_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    if feature_names is not None:
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        if all(feature in name_to_idx for feature in selected_features):
            indices = [name_to_idx[feature] for feature in selected_features]
            return X[:, indices], indices

    if selected_indices is not None and len(selected_indices) == len(selected_features):
        if max(selected_indices) >= X.shape[1] or min(selected_indices) < 0:
            raise ValueError(f"Selected feature indices out of bounds for input dimension {X.shape[1]}")
        return X[:, selected_indices], selected_indices

    missing = [feature for feature in selected_features if feature_names is None or feature not in set(feature_names)]
    raise ValueError(
        "Could not apply QGA feature selection. Missing feature names or indices for: "
        + ", ".join(missing[:5])
    )


def maybe_subsample(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=max_samples, replace=False)
    indices.sort()
    return X[indices], y[indices]


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


class FinalL1MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(fpr), cm


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    fpr, cm = compute_binary_fpr(y_true, y_pred)

    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr": fpr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }


def plot_loss_curves(history: list[dict[str, Any]], final_test_loss: float, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_loss, marker="s", linewidth=2, label="Validation Loss")
    plt.axhline(final_test_loss, linestyle="--", linewidth=2, label=f"Final Test Loss = {final_test_loss:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final L1 Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_validation_metrics(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    val_macro_f1 = [row["val_macro_f1"] for row in history]
    val_attack_recall = [row["val_attack_recall"] for row in history]
    val_fpr = [row["val_fpr"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_macro_f1, marker="o", linewidth=2, label="Validation Macro-F1")
    plt.plot(epochs, val_attack_recall, marker="s", linewidth=2, label="Validation Attack Recall")
    plt.plot(epochs, val_fpr, marker="^", linewidth=2, label="Validation FPR")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Final L1 Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    matrix = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Final L1 Test Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.yticks([0, 1], ["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_history_csv(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final L1 model and generate report curves.")
    parser.add_argument("--train", type=Path, default=Path("experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/train_scaled.npz"))
    parser.add_argument("--val", type=Path, default=Path("experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/val_scaled.npz"))
    parser.add_argument("--test", type=Path, default=Path("experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/test_scaled.npz"))
    parser.add_argument("--feature-names", type=Path, default=REPO_DEFAULT_FEATURE_NAMES)
    parser.add_argument("--qga-dir", type=Path, default=Path("experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/qi-fl-ids-iot-final/outputs/final_report_curves"))
    parser.add_argument("--use-qga-mask", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, feature_names = load_npz_dataset(args.train, args.feature_names)
    X_val, y_val, feature_names_val = load_npz_dataset(args.val, args.feature_names)
    X_test, y_test, feature_names_test = load_npz_dataset(args.test, args.feature_names)

    if feature_names != feature_names_val or feature_names != feature_names_test:
        raise ValueError("Feature names mismatch between train, validation, and test datasets.")

    metadata: dict[str, Any] = {
        "task": "l1_binary",
        "use_qga_mask": bool(args.use_qga_mask),
        "input_dim_original": int(X_train.shape[1]),
        "device": str(device),
        "seed": int(args.seed),
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "test_used_for_final_evaluation_only": True,
    }

    if args.use_qga_mask:
        selected_features, selected_indices, decision = load_qga_selected_features(args.qga_dir)
        X_train, applied_indices = apply_feature_selection(X_train, feature_names, selected_features, selected_indices)
        X_val, _ = apply_feature_selection(X_val, feature_names, selected_features, selected_indices)
        X_test, _ = apply_feature_selection(X_test, feature_names, selected_features, selected_indices)
        metadata.update(
            {
                "selected_mask_id": decision.get("selected_mask_id"),
                "selected_features_count": len(selected_features),
                "selected_features": selected_features,
                "selected_indices": applied_indices,
                "calibration_decision_used": True,
                "selected_mask_source": "final_selected_mask",
            }
        )
    else:
        metadata.update(
            {
                "selected_features_count": int(X_train.shape[1]),
                "calibration_decision_used": False,
                "selected_mask_source": None,
            }
        )

    X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, args.seed)
    X_val, y_val = maybe_subsample(X_val, y_val, args.max_val_samples, args.seed + 1)
    X_test, y_test = maybe_subsample(X_test, y_test, args.max_test_samples, args.seed + 2)

    input_dim = int(X_train.shape[1])
    metadata["input_dim_selected"] = input_dim
    metadata["model_architecture"] = f"{input_dim} -> 128 -> 64 -> 2"
    metadata["train_rows"] = int(len(y_train))
    metadata["val_rows"] = int(len(y_val))
    metadata["test_rows"] = int(len(y_test))

    train_loader = build_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = build_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = build_loader(X_test, y_test, args.batch_size, shuffle=False)

    model = FinalL1MLP(input_dim=input_dim, dropout=args.dropout).to(device)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * 2.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size

        train_loss = total_train_loss / max(total_train_samples, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_attack_recall": float(val_metrics["attack_recall"]),
            "val_fpr": float(val_metrics["fpr"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_attack_recall={val_metrics['attack_recall']:.4f} "
            f"val_fpr={val_metrics['fpr']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was captured.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    test_metrics.update(
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_macro_f1),
            "model_architecture": metadata["model_architecture"],
            "metadata": metadata,
        }
    )

    torch.save(model.state_dict(), args.output_dir / "best_model.pth")
    save_history_csv(history, args.output_dir / "training_history.csv")
    (args.output_dir / "final_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_loss_curves(history, float(test_metrics["loss"]), args.output_dir / "01_loss_curves.png")
    plot_validation_metrics(history, args.output_dir / "02_validation_metrics.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], args.output_dir / "03_test_confusion_matrix.png")

    print("\nFinal test evaluation:")
    print(json.dumps({key: value for key, value in test_metrics.items() if key != "confusion_matrix"}, indent=2))
    print("\nArtifacts generated:")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "best_epoch": best_epoch,
                "best_val_macro_f1": float(best_val_macro_f1),
                "final_test_loss": float(test_metrics["loss"]),
                "final_test_macro_f1": float(test_metrics["macro_f1"]),
                "final_test_attack_recall": float(test_metrics["attack_recall"]),
                "final_test_fpr": float(test_metrics["fpr"]),
                "final_test_accuracy": float(test_metrics["accuracy"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
