"""
run_hierarchical_pipeline_inference_v2.py
========================================

Production-style hierarchical inference with monitoring and evaluation.

What it does
------------
1. Loads the test CSV
2. Runs Level 1: benign vs attack
3. Runs Level 2: family classifier for attack samples
4. Runs Level 3: family-specific subtype classifier when needed
5. Saves:
   - final predictions CSV
   - summary JSON
   - level-wise metrics CSV
   - confusion matrix CSV / PNG
   - classification report CSV
"""

import json
import time
import pickle
from collections import Counter
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

LABEL_COL = "label"

ROOT_MODELS = Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3_models")
ROOT_DATA = Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3")
INPUT_CSV = Path(r"E:\dataset\processed_merged_full\minority_balancing_v3\training_ready\test.csv")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\hierarchical_final_inference_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL1_DIR = ROOT_MODELS / "level1_binary"
LEVEL2_DIR = ROOT_MODELS / "level2_family"
LEVEL3_MODELS_DIR = ROOT_MODELS / "level3_family_submodels"
LEVEL3_DATA_DIR = ROOT_DATA / "level3_family_submodels"
FAMILY_MAPPING_PATH = ROOT_DATA / "family_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_header(title: str):
    print("\\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


class SmallMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_dir: Path):
    with open(model_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open(model_dir / "label_mapping.pkl", "rb") as f:
        label_map = pickle.load(f)

    model = SmallMLP(len(feature_names), len(label_map))
    state = torch.load(model_dir / "mlp_model_state.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, feature_names, label_map


def predict(model, X: np.ndarray):
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        logits = model(xb)
        return torch.argmax(logits, dim=1).cpu().numpy()


def save_confusion(cm, labels, csv_path: Path, png_path: Path, title: str):
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(csv_path)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    t0 = time.time()

    print_header("LOAD INPUT")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print("Input CSV:", INPUT_CSV, flush=True)
    print("Shape:", df.shape, flush=True)

    if not FAMILY_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Missing family mapping: {FAMILY_MAPPING_PATH}")
    with open(FAMILY_MAPPING_PATH, "r", encoding="utf-8") as f:
        family_mapping = json.load(f)

    print_header("LOAD LEVEL 1")
    l1_model, l1_features, _ = load_model(LEVEL1_DIR)
    X1 = df[l1_features].to_numpy(dtype=np.float32)
    pred_l1 = predict(l1_model, X1)
    print("Level1 prediction distribution:", dict(Counter(pred_l1)), flush=True)

    print_header("LOAD LEVEL 2")
    l2_model, l2_features, l2_map = load_model(LEVEL2_DIR)
    family_id_to_name = {v: k for k, v in l2_map.items()}
    print("Family labels:", family_id_to_name, flush=True)

    predictions = []
    predicted_family = []
    level3_loaded = {}

    print_header("RUN HIERARCHICAL INFERENCE")
    for idx, row in df.iterrows():
        if idx % 50000 == 0 and idx > 0:
            print(f"Processed {idx:,} rows...", flush=True)

        if int(pred_l1[idx]) == 0:
            predictions.append("BENIGN")
            predicted_family.append("benign")
            continue

        x2 = row[l2_features].to_numpy(dtype=np.float32)[None, :]
        fam_id = int(predict(l2_model, x2)[0])
        family = family_id_to_name[fam_id]
        predicted_family.append(family)

        sub_data_dir = LEVEL3_DATA_DIR / family
        sub_model_dir = LEVEL3_MODELS_DIR / family

        with open(sub_data_dir / "label_mapping.pkl", "rb") as f:
            sub_map = pickle.load(f)

        if len(sub_map) == 1:
            predictions.append(next(iter(sub_map.keys())))
            continue

        if family not in level3_loaded:
            level3_loaded[family] = load_model(sub_model_dir)

        sub_model, sub_features, sub_loaded_map = level3_loaded[family]
        subtype_id_to_name = {v: k for k, v in sub_loaded_map.items()}
        x3 = row[sub_features].to_numpy(dtype=np.float32)[None, :]
        sub_id = int(predict(sub_model, x3)[0])
        predictions.append(subtype_id_to_name[sub_id])

    out = df.copy()
    out["predicted_family"] = predicted_family
    out["predicted_label"] = predictions
    pred_csv = OUTPUT_DIR / "hierarchical_predictions.csv"
    out.to_csv(pred_csv, index=False)

    print_header("PREDICTION DISTRIBUTIONS")
    print("Predicted families:", dict(Counter(predicted_family)), flush=True)
    print("Predicted labels top 20:", dict(Counter(predictions).most_common(20)), flush=True)

    summary = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUTPUT_DIR),
        "n_rows": int(len(out)),
        "device": str(DEVICE),
        "prediction_family_distribution": dict(Counter(predicted_family)),
        "prediction_label_distribution_top20": dict(Counter(predictions).most_common(20)),
    }

    level_metrics_rows = []

    if LABEL_COL in out.columns:
        print_header("EVALUATION")

        final_metrics = compute_metrics(out[LABEL_COL], out["predicted_label"])
        print("Final metrics:", final_metrics, flush=True)
        for k, v in final_metrics.items():
            level_metrics_rows.append({"level": "final_34class", "metric": k, "value": v})

        true_binary = (out[LABEL_COL] != "BENIGN").astype(int).to_numpy()
        level1_metrics = compute_metrics(true_binary, pred_l1)
        print("Level1 metrics:", level1_metrics, flush=True)
        for k, v in level1_metrics.items():
            level_metrics_rows.append({"level": "level1_binary", "metric": k, "value": v})

        attack_mask = out[LABEL_COL] != "BENIGN"
        true_families = out.loc[attack_mask, LABEL_COL].map(family_mapping)
        pred_families = out.loc[attack_mask, "predicted_family"]

        level2_metrics = compute_metrics(true_families, pred_families)
        print("Level2 metrics:", level2_metrics, flush=True)
        for k, v in level2_metrics.items():
            level_metrics_rows.append({"level": "level2_family", "metric": k, "value": v})

        pd.DataFrame(level_metrics_rows).to_csv(OUTPUT_DIR / "level_metrics.csv", index=False)

        final_report = pd.DataFrame(
            classification_report(
                out[LABEL_COL],
                out["predicted_label"],
                output_dict=True,
                zero_division=0,
            )
        ).T
        final_report.to_csv(OUTPUT_DIR / "classification_report_final.csv")

        labels_sorted = sorted(out[LABEL_COL].unique().tolist())
        cm_final = confusion_matrix(out[LABEL_COL], out["predicted_label"], labels=labels_sorted)
        save_confusion(
            cm_final,
            labels_sorted,
            OUTPUT_DIR / "confusion_matrix_final.csv",
            OUTPUT_DIR / "confusion_matrix_final.png",
            "Final Hierarchical 34-Class Confusion Matrix",
        )

        cm_l1 = confusion_matrix(true_binary, pred_l1, labels=[0, 1])
        save_confusion(
            cm_l1,
            ["benign", "attack"],
            OUTPUT_DIR / "confusion_matrix_level1.csv",
            OUTPUT_DIR / "confusion_matrix_level1.png",
            "Level 1 Binary Confusion Matrix",
        )

        family_labels = sorted(true_families.unique().tolist())
        cm_l2 = confusion_matrix(true_families, pred_families, labels=family_labels)
        save_confusion(
            cm_l2,
            family_labels,
            OUTPUT_DIR / "confusion_matrix_level2.csv",
            OUTPUT_DIR / "confusion_matrix_level2.png",
            "Level 2 Family Confusion Matrix",
        )

        summary["final_metrics"] = final_metrics
        summary["level1_metrics"] = level1_metrics
        summary["level2_metrics"] = level2_metrics

    summary["runtime_sec"] = float(time.time() - t0)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_header("DONE")
    print("Predictions saved to:", pred_csv, flush=True)
    print("Summary saved to:", OUTPUT_DIR / "summary.json", flush=True)


if __name__ == "__main__":
    main()
