"""
mlp_b_pytorch_earlystop.py
==========================

Centralized baseline for CICIoT2023 using PyTorch.

Architecture B
--------------
Input(32) -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.30)
          -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.20)
          -> Output(34)

Why this script
---------------
This is the stronger compromise architecture for:
- centralized baseline quality
- later FL compatibility
- still-reasonable model size for edge-oriented discussion

Main improvements
-----------------
- PyTorch training
- GPU support if CUDA is available
- focal loss for class imbalance
- class weights
- validation monitoring
- early stopping before overfitting/divergence
- checkpointing of the best model
- saved metrics, confusion matrix, scaler, mappings, and feature names
"""

import os
import gc
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
LABEL_COL = "label"

DATA_PATH = Path(r"E:\dataset\processed_merged_full\augmentation_controlled\dataset_34classes_augmented_controlled.csv")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\mlp_b_pytorch_results")

DROP_DUPLICATES = True
SAVE_PLOTS = True

TEST_SIZE = 0.15
VAL_SIZE_FROM_REMAINING = 0.17647058823529413

BATCH_SIZE = 1024
MAX_EPOCHS = 70
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0

HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT_1 = 0.30
DROPOUT_2 = 0.20

FOCAL_GAMMA = 2.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MLPArchitectureB(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_2),
            nn.Linear(HIDDEN_2, num_classes),
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

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

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

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(running_loss / max(n_samples, 1))
    return metrics, y_true, y_pred


def main():
    print_header("CONFIGURATION")
    print("Dataset:", DATA_PATH, flush=True)
    print("Output dir:", OUTPUT_DIR, flush=True)
    print("Device:", DEVICE, flush=True)
    print("Architecture B: 32 -> 128 -> 64 -> 34", flush=True)
    print("Max epochs:", MAX_EPOCHS, flush=True)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE, flush=True)

    print_header("LOAD DATASET")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape, flush=True)
    show_df("HEAD", df, n=5)

    assert LABEL_COL in df.columns, f"Missing label column: {LABEL_COL}"

    print_header("SANITY CHECKS")
    print(df.info(), flush=True)

    missing = df.isna().sum().sort_values(ascending=False)
    show_df("Missing values (top 20)", missing, n=20)

    duplicate_count = int(df.duplicated().sum())
    print("\nDuplicate rows:", duplicate_count, flush=True)

    if DROP_DUPLICATES and duplicate_count > 0:
        print("Dropping duplicates...", flush=True)
        df = df.drop_duplicates().reset_index(drop=True)
        print("Shape after drop_duplicates:", df.shape, flush=True)

    class_counts = df[LABEL_COL].value_counts()
    show_df("Class distribution", class_counts, n=40)
    print("\nNumber of classes:", len(class_counts), flush=True)
    print("Imbalance ratio:", round(class_counts.max() / class_counts.min(), 2), flush=True)

    print_header("FEATURES AND LABELS")
    excluded_cols = [LABEL_COL]
    if "label_id_34" in df.columns:
        excluded_cols.append("label_id_34")

    feature_cols = [c for c in df.columns if c not in excluded_cols]
    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy()

    print("Feature count:", len(feature_cols), flush=True)
    print("First features:", feature_cols[:10], flush=True)
    print("X shape:", X.shape, flush=True)
    print("y shape:", y.shape, flush=True)

    print_header("LABEL ENCODING")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    label_to_id = {label: int(i) for i, label in enumerate(label_encoder.classes_)}
    id_to_label = {int(i): label for i, label in enumerate(label_encoder.classes_)}

    show_df(
        "Encoded labels",
        pd.DataFrame({"label": list(label_to_id.keys()), "label_id": list(label_to_id.values())}),
        n=40,
    )

    print_header("TRAIN / VAL / TEST SPLIT")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=(TEST_SIZE + 0.15),
        random_state=SEED,
        stratify=y_encoded,
    )

    del X, y, y_encoded
    gc.collect()

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    del X_temp, y_temp
    gc.collect()

    print("Train:", X_train.shape, y_train.shape, flush=True)
    print("Val  :", X_val.shape, y_val.shape, flush=True)
    print("Test :", X_test.shape, y_test.shape, flush=True)

    print_header("CLASS WEIGHTS")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    class_weight_df = pd.DataFrame({
        "class_id": list(class_weights.keys()),
        "weight": list(class_weights.values()),
        "label": [id_to_label[i] for i in class_weights.keys()]
    }).sort_values("weight", ascending=False)

    show_df("Top class weights", class_weight_df, n=15)

    print_header("SCALING")
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    del X_train, X_val, X_test
    gc.collect()

    print("Scaled train:", X_train_scaled.shape, X_train_scaled.dtype, flush=True)
    print("Scaled val  :", X_val_scaled.shape, X_val_scaled.dtype, flush=True)
    print("Scaled test :", X_test_scaled.shape, X_test_scaled.dtype, flush=True)

    print_header("BUILD TENSORS AND LOADERS")
    X_train_tensor = torch.from_numpy(X_train_scaled)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    X_val_tensor = torch.from_numpy(X_val_scaled)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))
    X_test_tensor = torch.from_numpy(X_test_scaled)
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

    del X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    model = MLPArchitectureB(input_dim=len(feature_cols), num_classes=len(label_to_id)).to(DEVICE)

    alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_state = None
    best_epoch = -1
    best_val_macro_f1 = -math.inf
    patience_counter = 0

    print_header("TRAINING")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)

        row = {
            "epoch": epoch,
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
    report_dict = classification_report(
        y_test_true,
        y_test_pred,
        target_names=label_encoder.classes_,
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
            "Confusion Matrix — PyTorch MLP B (32 -> 128 -> 64 -> 34)"
        )
        print("Saved confusion matrix plot:", OUTPUT_DIR / "confusion_matrix_test.png", flush=True)

    print_header("SAVE RESULTS")
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame(val_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "val_metrics.csv", index=False)
    pd.DataFrame(test_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report_test.csv")
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / "confusion_matrix_test.csv", index=False)

    torch.save(model.state_dict(), OUTPUT_DIR / "mlp_model_state.pt")

    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "label_mapping.pkl", "wb") as f:
        pickle.dump(label_to_id, f)
    with open(OUTPUT_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(OUTPUT_DIR / "class_weights.pkl", "wb") as f:
        pickle.dump(class_weights, f)

    summary = {
        "dataset_path": str(DATA_PATH),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(label_to_id)),
        "model": "PyTorch MLP",
        "architecture": "32 -> 128 -> 64 -> 34",
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "focal_gamma": FOCAL_GAMMA,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
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

    print("Saved files:", flush=True)
    for p in sorted(OUTPUT_DIR.iterdir()):
        print("-", p.name, flush=True)

    print_header("DONE")


if __name__ == "__main__":
    main()
