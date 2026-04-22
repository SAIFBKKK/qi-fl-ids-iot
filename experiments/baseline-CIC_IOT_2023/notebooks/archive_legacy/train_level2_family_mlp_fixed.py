"""
train_level2_family_mlp_fixed.py
================================

Fixed Level 2 hierarchical IDS training script.

Fix:
- remaps family labels from {1..8} to contiguous ids {0..7}
- validates target ranges before training
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
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42

INPUT_DIR = Path(r"E:\dataset\processed_merged_full\hierarchical_prepared\family")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\hierarchical_models\level2_family_fixed")

TRAIN_FILE = INPUT_DIR / "train.npz"
VAL_FILE = INPUT_DIR / "val.npz"
TEST_FILE = INPUT_DIR / "test.npz"
LABEL_MAP_FILE = INPUT_DIR / "label_mapping.pkl"
SUMMARY_FILE = INPUT_DIR / "summary.json"

BATCH_SIZE = 2048
MAX_EPOCHS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0

HIDDEN_1 = 64
DROPOUT_1 = 0.20

CLASS_WEIGHT_MIN = 1.0
CLASS_WEIGHT_MAX = 8.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert TRAIN_FILE.exists(), f"Missing {TRAIN_FILE}"
assert VAL_FILE.exists(), f"Missing {VAL_FILE}"
assert TEST_FILE.exists(), f"Missing {TEST_FILE}"
assert LABEL_MAP_FILE.exists(), f"Missing {LABEL_MAP_FILE}"
assert SUMMARY_FILE.exists(), f"Missing {SUMMARY_FILE}"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)


def show_df(title, obj, n=10):
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


class FamilyMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
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


def save_confusion_matrix(cm, filename, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
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


def remap_labels_to_contiguous(y_train, y_val, y_test, original_mapping):
    unique_ids = sorted(original_mapping.values())
    contiguous_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    y_train_new = np.vectorize(contiguous_id_map.get)(y_train).astype(np.int64)
    y_val_new = np.vectorize(contiguous_id_map.get)(y_val).astype(np.int64)
    y_test_new = np.vectorize(contiguous_id_map.get)(y_test).astype(np.int64)

    new_label_mapping = {}
    for label_name, old_id in original_mapping.items():
        new_label_mapping[label_name] = contiguous_id_map[old_id]

    return y_train_new, y_val_new, y_test_new, new_label_mapping, contiguous_id_map


def validate_targets(y, num_classes, split_name):
    y_min = int(np.min(y))
    y_max = int(np.max(y))
    unique = np.unique(y)
    print(f"{split_name}: min={y_min}, max={y_max}, unique={unique.tolist()}", flush=True)
    assert y_min >= 0, f"{split_name} has negative labels"
    assert y_max < num_classes, f"{split_name} has label {y_max} but num_classes={num_classes}"


def main():
    print_header("CONFIGURATION")
    print("Input dir:", INPUT_DIR, flush=True)
    print("Output dir:", OUTPUT_DIR, flush=True)
    print("Device:", DEVICE, flush=True)
    print("Architecture:", "32 -> 64 -> 8", flush=True)
    print("Max epochs:", MAX_EPOCHS, flush=True)
    print("Learning rate:", LEARNING_RATE, flush=True)

    print_header("LOAD ARTIFACTS")
    with open(LABEL_MAP_FILE, "rb") as f:
        original_label_mapping = pickle.load(f)

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data_summary = json.load(f)

    train_npz = np.load(TRAIN_FILE)
    val_npz = np.load(VAL_FILE)
    test_npz = np.load(TEST_FILE)

    X_train = train_npz["X"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    X_val = val_npz["X"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)
    X_test = test_npz["X"].astype(np.float32)
    y_test = test_npz["y"].astype(np.int64)

    print("Original label mapping:", original_label_mapping, flush=True)
    print("Summary:", data_summary, flush=True)

    print_header("REMAP LABELS TO CONTIGUOUS IDS")
    y_train, y_val, y_test, label_mapping, contiguous_id_map = remap_labels_to_contiguous(
        y_train, y_val, y_test, original_label_mapping
    )
    print("New contiguous label mapping:", label_mapping, flush=True)
    print("Contiguous id map:", contiguous_id_map, flush=True)

    num_classes = len(label_mapping)

    print_header("VALIDATE TARGET RANGES")
    validate_targets(y_train, num_classes, "train")
    validate_targets(y_val, num_classes, "val")
    validate_targets(y_test, num_classes, "test")

    print_header("DATA SHAPES")
    print("Train:", X_train.shape, y_train.shape, flush=True)
    print("Val  :", X_val.shape, y_val.shape, flush=True)
    print("Test :", X_test.shape, y_test.shape, flush=True)

    print_header("CLASS WEIGHTS")
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    clipped_weights = np.clip(raw_weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)

    cw_df = pd.DataFrame({
        "class_id": classes,
        "raw_weight": raw_weights,
        "clipped_weight": clipped_weights,
    })
    show_df("Family class weights", cw_df, n=20)

    print_header("BUILD TENSORS AND LOADERS")
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    print("Train batches:", len(train_loader), flush=True)
    print("Val batches  :", len(val_loader), flush=True)
    print("Test batches :", len(test_loader), flush=True)

    print_header("BUILD MODEL")
    model = FamilyMLP(input_dim=X_train.shape[1], num_classes=num_classes).to(DEVICE)
    weight_tensor = torch.tensor(clipped_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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
    target_names = [k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    report_dict = classification_report(y_test_true, y_test_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    show_df("Classification report", report_df, n=20)

    print_header("CONFUSION MATRIX")
    cm = confusion_matrix(y_test_true, y_test_pred)
    save_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix_test.png", "Level 2 Family Confusion Matrix")
    print("Saved confusion matrix plot:", OUTPUT_DIR / "confusion_matrix_test.png", flush=True)

    print_header("SAVE RESULTS")
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame(val_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "val_metrics.csv", index=False)
    pd.DataFrame(test_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report_test.csv")
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / "confusion_matrix_test.csv", index=False)

    torch.save(model.state_dict(), OUTPUT_DIR / "mlp_model_state.pt")
    with open(OUTPUT_DIR / "label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)
    with open(OUTPUT_DIR / "original_label_mapping.pkl", "wb") as f:
        pickle.dump(original_label_mapping, f)

    run_summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "class_weight_min": CLASS_WEIGHT_MIN,
        "class_weight_max": CLASS_WEIGHT_MAX,
        "training_time_sec": float(total_time),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_classes": int(num_classes),
        "label_mapping": label_mapping,
        "original_label_mapping": original_label_mapping,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Level 2 Family Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Level 2 Family Validation Macro F1")
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
