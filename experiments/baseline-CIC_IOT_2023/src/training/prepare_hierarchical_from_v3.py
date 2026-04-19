"""
prepare_hierarchical_from_v3.py
===============================

Prepare final hierarchical datasets from the balanced v3 dataset.
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_COL = "label"

DATASET_PATH = Path(r"E:\dataset\processed_merged_full\minority_balancing_v3\dataset_34classes_balanced_v3.csv")
OUTPUT_DIR = Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_MAPPING = {
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
FEATURE_EXCLUDE = {"label", "label_id_34", "binary_label", "family", "family_id", "subtype_id"}

def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def split_df(df: pd.DataFrame, stratify_col: str):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[stratify_col], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[stratify_col], random_state=SEED)
    return train_df, val_df, test_df

def save_split(train_df, val_df, test_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading:", DATASET_PATH, flush=True)
    df = pd.read_csv(DATASET_PATH)
    print("Shape:", df.shape, flush=True)

    df["binary_label"] = (df[LABEL_COL] != "BENIGN").astype(int)
    df["family"] = df[LABEL_COL].map(FAMILY_MAPPING)

    missing = df[df["family"].isna()][LABEL_COL].unique().tolist()
    if missing:
        raise ValueError(f"Missing family mapping for: {missing}")

    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    save_pickle(feature_cols, OUTPUT_DIR / "feature_names.pkl")
    save_json(FAMILY_MAPPING, OUTPUT_DIR / "family_mapping.json")

    level1_dir = OUTPUT_DIR / "level1_binary"
    train_df, val_df, test_df = split_df(df.copy(), "binary_label")
    save_split(train_df, val_df, test_df, level1_dir)
    save_pickle({"benign": 0, "attack": 1}, level1_dir / "label_mapping.pkl")

    attack_df = df[df[LABEL_COL] != "BENIGN"].copy()
    family_names = sorted(attack_df["family"].unique())
    family_to_id = {name: idx for idx, name in enumerate(family_names)}
    attack_df["family_id"] = attack_df["family"].map(family_to_id)

    level2_dir = OUTPUT_DIR / "level2_family"
    train_df, val_df, test_df = split_df(attack_df.copy(), "family_id")
    save_split(train_df, val_df, test_df, level2_dir)
    save_pickle(family_to_id, level2_dir / "label_mapping.pkl")

    level3_root = OUTPUT_DIR / "level3_family_submodels"
    level3_root.mkdir(parents=True, exist_ok=True)

    family_summaries = {}
    for family in family_names:
        family_df = attack_df[attack_df["family"] == family].copy()
        labels = sorted(family_df[LABEL_COL].unique().tolist())
        out_dir = level3_root / family
        out_dir.mkdir(parents=True, exist_ok=True)

        label_map = {label: idx for idx, label in enumerate(labels)}
        save_pickle(label_map, out_dir / "label_mapping.pkl")

        if len(labels) < 2:
            save_json({"family": family, "n_rows": int(len(family_df)), "n_classes": int(len(labels)), "labels": labels, "note": "single-class family"}, out_dir / "summary.json")
            family_summaries[family] = {"n_classes": 1, "labels": labels}
            continue

        family_df["subtype_id"] = family_df[LABEL_COL].map(label_map)
        train_df, val_df, test_df = split_df(family_df.copy(), "subtype_id")
        save_split(train_df, val_df, test_df, out_dir)
        save_json({"family": family, "n_rows": int(len(family_df)), "n_classes": int(len(labels)), "labels": labels}, out_dir / "summary.json")
        family_summaries[family] = {"n_classes": len(labels), "labels": labels}

    save_json({"dataset_path": str(DATASET_PATH), "output_dir": str(OUTPUT_DIR), "n_rows": int(len(df)), "n_features": int(len(feature_cols)), "n_level2_families": int(len(family_names)), "families": family_summaries}, OUTPUT_DIR / "hierarchy_summary.json")
    print("Done:", OUTPUT_DIR, flush=True)

if __name__ == "__main__":
    main()
