"""
prepare_hierarchical_iot_ids.py
===============================

Prepare hierarchical datasets and artifacts for the CICIoT2023 project.

Uses the existing stable flat training artifacts from:
E:\dataset\processed_merged_full\stable_flat_mlp_a_results

Creates:
- binary level dataset (benign vs attack)
- family level dataset (attack families)
- family-specific fine-grained datasets

Outputs under:
E:\dataset\processed_merged_full\hierarchical_prepared
"""

import gc
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEED = 42
LABEL_COL = "label"

DATA_PATH = Path(r"E:\dataset\processed_merged_full\augmentation_controlled\dataset_34classes_augmented_controlled.csv")
PREVIOUS_ARTIFACTS_DIR = Path(r"E:\dataset\processed_merged_full\stable_flat_mlp_a_results")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\hierarchical_prepared")

DROP_DUPLICATES = True
TEMP_SIZE = 0.30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


FAMILY_MAPPING_34_TO_8 = {
    "BENIGN": "benign",
    "DDOS-ACK_FRAGMENTATION": "ddos",
    "DDOS-HTTP_FLOOD": "ddos",
    "DDOS-ICMP_FLOOD": "ddos",
    "DDOS-ICMP_FRAGMENTATION": "ddos",
    "DDOS-PSHACK_FLOOD": "ddos",
    "DDOS-RSTFINFLOOD": "ddos",
    "DDOS-SLOWLORIS": "ddos",
    "DDOS-SYNONYMOUSIP_FLOOD": "ddos",
    "DDOS-SYN_FLOOD": "ddos",
    "DDOS-TCP_FLOOD": "ddos",
    "DDOS-UDP_FLOOD": "ddos",
    "DDOS-UDP_FRAGMENTATION": "ddos",
    "DOS-HTTP_FLOOD": "dos",
    "DOS-SYN_FLOOD": "dos",
    "DOS-TCP_FLOOD": "dos",
    "DOS-UDP_FLOOD": "dos",
    "MIRAI-GREETH_FLOOD": "mirai",
    "MIRAI-GREIP_FLOOD": "mirai",
    "MIRAI-UDPPLAIN": "mirai",
    "RECON-HOSTDISCOVERY": "recon",
    "RECON-OSSCAN": "recon",
    "RECON-PINGSWEEP": "recon",
    "RECON-PORTSCAN": "recon",
    "VULNERABILITYSCAN": "recon",
    "BROWSERHIJACKING": "web",
    "COMMANDINJECTION": "web",
    "SQLINJECTION": "web",
    "UPLOADING_ATTACK": "web",
    "XSS": "web",
    "DNS_SPOOFING": "spoofing",
    "MITM-ARPSPOOFING": "spoofing",
    "DICTIONARYBRUTEFORCE": "bruteforce",
    "BACKDOOR_MALWARE": "malware",
}

FAMILY_TO_ID = {
    "benign": 0,
    "ddos": 1,
    "dos": 2,
    "mirai": 3,
    "recon": 4,
    "web": 5,
    "spoofing": 6,
    "bruteforce": 7,
    "malware": 8,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def choose_feature_columns(df):
    feature_path = PREVIOUS_ARTIFACTS_DIR / "feature_names.pkl"
    if feature_path.exists():
        with open(feature_path, "rb") as f:
            feature_cols = pickle.load(f)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns from previous artifacts: {missing}")
        return feature_cols

    excluded = {
        "label", "label_id_34", "binary_label", "attack_family",
        "attack_family_id", "family_label", "family_label_id"
    }
    return [c for c in df.columns if c not in excluded]


def split_three_way(X_df, y, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df, y, test_size=TEMP_SIZE, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_csv(base_dir, X_train, X_val, X_test, y_train, y_val, y_test, target_name):
    base_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target_name] = y_train
    train_df.to_csv(base_dir / "train.csv", index=False)

    val_df = X_val.copy()
    val_df[target_name] = y_val
    val_df.to_csv(base_dir / "val.csv", index=False)

    test_df = X_test.copy()
    test_df[target_name] = y_test
    test_df.to_csv(base_dir / "test.csv", index=False)


def save_scaled_npz(base_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    np.savez_compressed(base_dir / "train.npz", X=X_train_scaled, y=np.asarray(y_train, dtype=np.int64))
    np.savez_compressed(base_dir / "val.npz", X=X_val_scaled, y=np.asarray(y_val, dtype=np.int64))
    np.savez_compressed(base_dir / "test.npz", X=X_test_scaled, y=np.asarray(y_test, dtype=np.int64))

    save_pickle(scaler, base_dir / "scaler.pkl")


def prepare_level_dataset(df, feature_cols, target_col, label_mapping, base_dir):
    X_df = df[feature_cols].copy()
    y = df[target_col].to_numpy()

    X_train, X_val, X_test, y_train, y_val, y_test = split_three_way(X_df, y, random_state=SEED)

    save_split_csv(base_dir, X_train, X_val, X_test, y_train, y_val, y_test, target_col)
    save_scaled_npz(base_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    save_pickle(label_mapping, base_dir / "label_mapping.pkl")

    summary = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(np.unique(y))),
        "target_col": target_col,
    }
    save_json(summary, base_dir / "summary.json")
    return summary


def prepare_family_submodels(df, feature_cols, base_dir):
    family_models_dir = base_dir / "family_models"
    family_models_dir.mkdir(parents=True, exist_ok=True)
    family_summaries = {}

    for family_name in sorted(df["attack_family"].unique()):
        if family_name == "benign":
            continue

        family_df = df[df["attack_family"] == family_name].copy()
        sublabels = sorted(family_df["label"].unique().tolist())

        family_dir = family_models_dir / family_name
        family_dir.mkdir(parents=True, exist_ok=True)

        label_mapping = {label: idx for idx, label in enumerate(sublabels)}
        family_df["family_label_id"] = family_df["label"].map(label_mapping)

        summary = {
            "family_name": family_name,
            "n_rows": int(len(family_df)),
            "n_subclasses": int(len(sublabels)),
            "subclasses": sublabels,
        }

        if len(sublabels) < 2:
            save_json(summary, family_dir / "summary.json")
            save_pickle(label_mapping, family_dir / "label_mapping.pkl")
            family_summaries[family_name] = summary
            continue

        X_df = family_df[feature_cols].copy()
        y = family_df["family_label_id"].to_numpy()

        X_train, X_val, X_test, y_train, y_val, y_test = split_three_way(X_df, y, random_state=SEED)

        save_split_csv(family_dir, X_train, X_val, X_test, y_train, y_val, y_test, "family_label_id")
        save_scaled_npz(family_dir, X_train, X_val, X_test, y_train, y_val, y_test)
        save_pickle(label_mapping, family_dir / "label_mapping.pkl")

        summary.update({
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feature_cols)),
        })
        save_json(summary, family_dir / "summary.json")
        family_summaries[family_name] = summary

    return family_summaries


def main():
    set_seed(SEED)

    print_header("LOAD DATASET")
    print("Dataset:", DATA_PATH, flush=True)
    print("Previous artifacts dir:", PREVIOUS_ARTIFACTS_DIR, flush=True)
    print("Output dir:", OUTPUT_DIR, flush=True)

    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape, flush=True)

    if DROP_DUPLICATES:
        dup_count = int(df.duplicated().sum())
        print("Duplicate rows:", dup_count, flush=True)
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            print("Shape after drop_duplicates:", df.shape, flush=True)

    print_header("BUILD HIERARCHICAL LABELS")
    df["binary_label"] = (df["label"] != "BENIGN").astype(int)
    df["attack_family"] = df["label"].map(FAMILY_MAPPING_34_TO_8)

    missing = df[df["attack_family"].isna()]["label"].unique().tolist()
    if missing:
        raise ValueError(f"Missing labels in family mapping: {missing}")

    df["attack_family_id"] = df["attack_family"].map(FAMILY_TO_ID)

    print("Binary label distribution:")
    print(df["binary_label"].value_counts().to_string(), flush=True)

    print("\nFamily distribution:")
    print(df["attack_family"].value_counts().to_string(), flush=True)

    print_header("FEATURE SELECTION")
    feature_cols = choose_feature_columns(df)
    print("Feature count:", len(feature_cols), flush=True)
    print("First features:", feature_cols[:10], flush=True)

    save_pickle(feature_cols, OUTPUT_DIR / "feature_names.pkl")
    save_json(FAMILY_MAPPING_34_TO_8, OUTPUT_DIR / "family_mapping.json")
    save_json(FAMILY_TO_ID, OUTPUT_DIR / "family_id_mapping.json")

    print_header("PREPARE LEVEL 1 - BINARY")
    binary_dir = OUTPUT_DIR / "binary"
    binary_summary = prepare_level_dataset(
        df=df,
        feature_cols=feature_cols,
        target_col="binary_label",
        label_mapping={"benign": 0, "attack": 1},
        base_dir=binary_dir,
    )
    print("Binary summary:", binary_summary, flush=True)

    print_header("PREPARE LEVEL 2 - FAMILY")
    family_df = df[df["label"] != "BENIGN"].copy()
    family_dir = OUTPUT_DIR / "family"
    family_summary = prepare_level_dataset(
        df=family_df,
        feature_cols=feature_cols,
        target_col="attack_family_id",
        label_mapping={k: v for k, v in FAMILY_TO_ID.items() if k != "benign"},
        base_dir=family_dir,
    )
    print("Family summary:", family_summary, flush=True)

    print_header("PREPARE LEVEL 3 - FAMILY-SPECIFIC SUBMODELS")
    family_submodel_summaries = prepare_family_submodels(family_df, feature_cols, OUTPUT_DIR)
    print("Prepared families:", list(family_submodel_summaries.keys()), flush=True)

    hierarchy_summary = {
        "dataset_path": str(DATA_PATH),
        "previous_artifacts_dir": str(PREVIOUS_ARTIFACTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "n_rows_final": int(len(df)),
        "n_features": int(len(feature_cols)),
        "binary_summary": binary_summary,
        "family_summary": family_summary,
        "family_submodels": family_submodel_summaries,
    }
    save_json(hierarchy_summary, OUTPUT_DIR / "hierarchy_summary.json")

    print_header("DONE")
    print("Files prepared successfully.", flush=True)
    print("Main output:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
