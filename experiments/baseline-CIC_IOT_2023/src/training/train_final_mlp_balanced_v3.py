"""
train_final_mlp_balanced_v3.py
==============================

Final centralized MLP training script for the balanced CICIoT2023 dataset.

Source dataset artifacts
------------------------
Prepared from:
    E:\dataset\processed_merged_full\minority_balancing_v3\training_ready

Expected files:
    - train.csv
    - val.csv
    - test.csv
    - label_mapping_34.pkl
    - class_weights_34.pkl
    - weighted_sampler_row_weights.npy

Model
-----
Stable lightweight MLP:
    Input(32) -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.20) -> Output(34)

Main upgrades vs older flat baseline
------------------------------------
- trains from the new balanced-v3 dataset artifacts
- uses WeightedRandomSampler for the training split
- uses precomputed class weights from training_ready
- uses ReduceLROnPlateau scheduler
- uses early stopping on validation macro-F1
- saves all metrics, artifacts, and plots
"""

import json
import math
import time
import copy
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")

SEED = 42
LABEL_COL = "label"

DATA_DIR = Path(r"E:\dataset\processed_merged_full\minority_balancing_v3\training_ready")
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
LABEL_MAP_PATH = DATA_DIR / "label_mapping_34.pkl"
CLASS_WEIGHTS_PATH = DATA_DIR / "class_weights_34.pkl"
SAMPLER_WEIGHTS_PATH = DATA_DIR / "weighted_sampler_row_weights.npy"

OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\final_mlp_balanced_v3_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = True
SAVE_MODEL = True

BATCH_SIZE = 2048
MAX_EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6
NUM_WORKERS = 0

HIDDEN_1 = 64
DROPOUT_1 = 0.20

assert TRAIN_PATH.exists(), f"Missing file: {TRAIN_PATH}"
assert VAL_PATH.exists(), f"Missing file: {VAL_PATH}"
assert TEST_PATH.exists(), f"Missing file: {TEST_PATH}"
assert LABEL_MAP_PATH.exists(), f"Missing file: {LABEL_MAP_PATH}"
assert CLASS_WEIGHTS_PATH.exists(), f"Missing file: {CLASS_WEIGHTS_PATH}"
assert SAMPLER_WEIGHTS_PATH.exists(), f"Missing file: {SAMPLER_WEIGHTS_PATH}"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)


def show_df(title: str, obj, n: int = 10) -> None:
    print(f"\n--- {title} ---", flush=True)
    if isinstance(obj, pd.DataFrame):
        print(obj.head(n).to_string(), flush=True)
    elif isinstance(obj, pd.Series):
        print(obj.head(n).to_string(), flush=True)
    else:
        print(obj, flush=True)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


class FinalBalancedMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(HIDDEN_1, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def save_confusion_matrix(cm, filename: Path, title: str):
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(running_loss / max(n_samples, 1))
    return metrics, y_true, y_pred


def load_split(csv_path: Path, label_to_id: dict):
    df = pd.read_csv(csv_path)
    assert LABEL_COL in df.columns, f"{csv_path} missing {LABEL_COL}"
    feature_cols = [c for c in df.columns if c not in [LABEL_COL, "label_id_34"]]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].map(label_to_id).to_numpy(dtype=np.int64)

    if np.any(pd.isna(y)):
        raise ValueError(f"Found unmapped labels in {csv_path}")

    return df, feature_cols, X, y


def main():
    print_header("CONFIGURATION")
    print("Data dir:", DATA_DIR, flush=True)
    print("Output dir:", OUTPUT_DIR, flush=True)
    print("Device:", DEVICE, flush=True)
    print("Architecture:", "32 -> 64 -> 34", flush=True)
    print("Loss:", "CrossEntropyLoss + precomputed class weights", flush=True)
    print("Sampler:", "WeightedRandomSampler", flush=True)
    print("Scheduler:", "ReduceLROnPlateau", flush=True)
    print("Max epochs:", MAX_EPOCHS, flush=True)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE, flush=True)
    print("Learning rate:", LEARNING_RATE, flush=True)

    print_header("LOAD ARTIFACTS")
    with open(LABEL_MAP_PATH, "rb") as f:
        label_to_id = pickle.load(f)

    with open(CLASS_WEIGHTS_PATH, "rb") as f:
        class_weights_dict = pickle.load(f)

    sampler_row_weights = np.load(SAMPLER_WEIGHTS_PATH).astype(np.float32)

    id_to_label = {int(v): k for k, v in label_to_id.items()}
    label_mapping_df = pd.DataFrame({
        "label": [id_to_label[i] for i in sorted(id_to_label.keys())],
        "label_id": sorted(id_to_label.keys()),
    })
    show_df("Label mapping", label_mapping_df, n=40)

    train_df, train_feature_cols, X_train, y_train = load_split(TRAIN_PATH, label_to_id)
    val_df, val_feature_cols, X_val, y_val = load_split(VAL_PATH, label_to_id)
    test_df, test_feature_cols, X_test, y_test = load_split(TEST_PATH, label_to_id)

    assert train_feature_cols == val_feature_cols == test_feature_cols, "Feature columns mismatch across splits"

    feature_cols = train_feature_cols

    print("Train:", X_train.shape, y_train.shape, flush=True)
    print("Val  :", X_val.shape, y_val.shape, flush=True)
    print("Test :", X_test.shape, y_test.shape, flush=True)
    print("Feature count:", len(feature_cols), flush=True)
    print("First features:", feature_cols[:10], flush=True)

    print_header("TRAIN CLASS DISTRIBUTION")
    show_df("Train class counts", train_df[LABEL_COL].value_counts(), n=40)

    print_header("CLASS WEIGHTS")
    class_weight_df = pd.DataFrame({
        "class_id": sorted(class_weights_dict.keys()),
        "weight": [class_weights_dict[i] for i in sorted(class_weights_dict.keys())],
        "label": [id_to_label[i] for i in sorted(class_weights_dict.keys())],
    }).sort_values("weight", ascending=False)
    show_df("Precomputed class weights", class_weight_df, n=40)

    print_header("SAMPLER WEIGHTS")
    print("Sampler weights shape:", sampler_row_weights.shape, flush=True)
    print("Sampler weights min/max:", float(sampler_row_weights.min()), float(sampler_row_weights.max()), flush=True)

    print_header("BUILD TENSORS AND LOADERS")
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sampler_row_weights),
        num_samples=len(sampler_row_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Train batches:", len(train_loader), flush=True)
    print("Val batches  :", len(val_loader), flush=True)
    print("Test batches :", len(test_loader), flush=True)

    print_header("BUILD MODEL")
    model = FinalBalancedMLP(input_dim=len(feature_cols), num_classes=len(label_to_id)).to(DEVICE)

    class_weight_tensor = torch.tensor(
        [class_weights_dict[i] for i in sorted(class_weights_dict.keys())],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    history = []
    best_state = None
    best_epoch = -1
    best_val_macro_f1 = -math.inf
    patience_counter = 0

    print_header("TRAINING")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "learning_rate": float(current_lr),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "epoch_time_sec": float(time.time() - epoch_start),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"lr={row['learning_rate']:.7f} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"val_loss={row['val_loss']:.6f} | "
            f"val_macro_f1={row['val_macro_f1']:.6f} | "
            f"val_recall={row['val_macro_recall']:.6f}",
            flush=True,
        )

        if row["val_macro_f1"] > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = row["val_macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model at epoch {epoch}", flush=True)
        else:
            patience_counter += 1
            print(f"  -> No improvement. patience={patience_counter}/{EARLY_STOPPING_PATIENCE}", flush=True)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}", flush=True)
            break

    total_time = time.time() - start_time
    print("Training time (sec):", round(total_time, 2), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Best validation macro-F1:", round(best_val_macro_f1, 6), flush=True)

    print_header("LOAD BEST CHECKPOINT")
    if best_state is not None:
        model.load_state_dict(best_state)

    print_header("FINAL VALIDATION EVALUATION")
    val_metrics, y_val_true, y_val_pred = evaluate_model(model, val_loader, criterion, DEVICE)
    show_df("Validation metrics", pd.DataFrame(val_metrics.items(), columns=["metric", "value"]), n=20)

    print_header("FINAL TEST EVALUATION")
    test_metrics, y_test_true, y_test_pred = evaluate_model(model, test_loader, criterion, DEVICE)
    show_df("Test metrics", pd.DataFrame(test_metrics.items(), columns=["metric", "value"]), n=20)

    print_header("CLASSIFICATION REPORT")
    target_names = [id_to_label[i] for i in range(len(id_to_label))]
    report_dict = classification_report(
        y_test_true,
        y_test_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    show_df("Classification report (head)", report_df, n=40)

    print_header("CONFUSION MATRIX")
    cm = confusion_matrix(y_test_true, y_test_pred)
    if SAVE_PLOTS:
        save_confusion_matrix(
            cm,
            OUTPUT_DIR / "confusion_matrix_test.png",
            "Confusion Matrix — Final Balanced MLP (32 -> 64 -> 34)"
        )
        print("Saved confusion matrix plot:", OUTPUT_DIR / "confusion_matrix_test.png", flush=True)

    print_header("SAVE RESULTS")
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame(val_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "val_metrics.csv", index=False)
    pd.DataFrame(test_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report_test.csv")
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / "confusion_matrix_test.csv", index=False)

    if SAVE_MODEL:
        torch.save(model.state_dict(), OUTPUT_DIR / "mlp_model_state.pt")

    with open(OUTPUT_DIR / "label_mapping.pkl", "wb") as f:
        pickle.dump(label_to_id, f)
    with open(OUTPUT_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(OUTPUT_DIR / "class_weights.pkl", "wb") as f:
        pickle.dump(class_weights_dict, f)

    summary = {
        "data_dir": str(DATA_DIR),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(label_to_id)),
        "model": "Final Balanced PyTorch MLP",
        "architecture": "32 -> 64 -> 34",
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "min_lr": MIN_LR,
        "training_time_sec": float(total_time),
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if SAVE_PLOTS:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Validation Macro F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "val_macro_f1_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(history_df["epoch"], history_df["learning_rate"], label="learning_rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "learning_rate_curve.png", dpi=150)
        plt.close()

    print("Saved files:", flush=True)
    for p in sorted(OUTPUT_DIR.iterdir()):
        print("-", p.name, flush=True)

    print_header("DONE")


if __name__ == "__main__":
    main()
