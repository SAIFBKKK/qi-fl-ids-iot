"""
baseline_mlp_tiny_a.py
======================

Baseline centralisé MLP — Architecture A
----------------------------------------
Architecture:
    32 -> 64 -> 34

Objectif
--------
Entraîner directement un MLP léger, cohérent avec un contexte IoT / FL,
sur le dataset CICIoT2023 nettoyé et augmenté.

Pipeline
--------
1. Chargement du dataset CSV
2. Sanity checks
3. Suppression optionnelle des doublons
4. Préparation X / y
5. Label encoding
6. Split stratifié train / val / test
7. Calcul et sauvegarde des class weights
8. Robust scaling
9. Entraînement MLP
10. Évaluation validation + test
11. Sauvegarde des artifacts et rapports

Notes
-----
- Ce script utilise scikit-learn MLPClassifier (CPU).
- Il n'utilise pas CUDA.
- Il est conçu comme baseline centralisé simple et propre avant FL.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import pickle
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
from sklearn.neural_network import MLPClassifier

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path(r"E:\dataset\processed_merged_full\augmentation_controlled\dataset_34classes_augmented_controlled.csv")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\baseline_mlp_a_results")
LABEL_COL = "label"

DROP_DUPLICATES = True
SAVE_PLOTS = True

MLP_HIDDEN_SIZES = (64,)
MLP_BATCH_SIZE = 256
MLP_LR = 1e-3
MLP_MAX_ITER = 30
MLP_ALPHA = 1e-4
EARLY_STOPPING = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


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


def save_confusion_matrix(cm, filename: Path, title: str):
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def build_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_SIZES,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size=MLP_BATCH_SIZE,
        learning_rate_init=MLP_LR,
        max_iter=MLP_MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=SEED,
        verbose=True,
    )


def main():
    print_header("CONFIGURATION")
    print("Dataset:", DATA_PATH, flush=True)
    print("Output dir:", OUTPUT_DIR, flush=True)
    print("Architecture A:", "32 -> 64 -> 34", flush=True)

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

    show_df("Encoded labels", pd.DataFrame({"label": list(label_to_id.keys()), "label_id": list(label_to_id.values())}), n=40)

    print_header("TRAIN / VAL / TEST SPLIT")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30, random_state=SEED, stratify=y_encoded
    )

    del X, y, y_encoded
    gc.collect()

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
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

    print_header("TRAIN MLP")
    model = build_mlp()
    model.fit(X_train_scaled, y_train)

    print_header("VALIDATION EVALUATION")
    y_val_pred = model.predict(X_val_scaled)
    val_metrics = compute_metrics(y_val, y_val_pred)
    show_df("Validation metrics", pd.DataFrame(val_metrics.items(), columns=["metric", "value"]), n=20)

    print_header("TEST EVALUATION")
    y_test_pred = model.predict(X_test_scaled)
    test_metrics = compute_metrics(y_test, y_test_pred)
    show_df("Test metrics", pd.DataFrame(test_metrics.items(), columns=["metric", "value"]), n=20)

    print_header("CLASSIFICATION REPORT")
    report_dict = classification_report(
        y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    show_df("Classification report (head)", report_df, n=40)

    print_header("CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_test_pred)
    if SAVE_PLOTS:
        save_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix_test.png", "Confusion Matrix — MLP Architecture A (32 -> 64 -> 34)")
        print("Saved confusion matrix plot:", OUTPUT_DIR / "confusion_matrix_test.png", flush=True)

    print_header("SAVE RESULTS")
    pd.DataFrame(val_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "val_metrics.csv", index=False)
    pd.DataFrame(test_metrics.items(), columns=["metric", "value"]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report_test.csv")
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / "confusion_matrix_test.csv", index=False)

    with open(OUTPUT_DIR / "mlp_model.pkl", "wb") as f:
        pickle.dump(model, f)
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
        "model": "MLP",
        "architecture": "32 -> 64 -> 34",
        "hidden_layer_sizes": list(MLP_HIDDEN_SIZES),
        "batch_size": MLP_BATCH_SIZE,
        "learning_rate": MLP_LR,
        "max_iter": MLP_MAX_ITER,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved files:", flush=True)
    for p in sorted(OUTPUT_DIR.iterdir()):
        print("-", p.name, flush=True)

    print_header("DONE")


if __name__ == "__main__":
    main()
