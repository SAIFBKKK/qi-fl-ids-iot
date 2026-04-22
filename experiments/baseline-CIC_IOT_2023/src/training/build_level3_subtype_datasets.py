"""
Build the Level-3 subtype datasets from preprocessing exports.

This utility was extracted from `baseline-CIC_IOT_2023_v2` during cleanup so
the kept `baseline-CIC_IOT_2023` experiment retains the useful dataset-builder
logic before the donor folder is retired.

Expected input files under the experiment root:
    post_balancing_preprocessing_FINAL_base_balanced_only/exports/
        - train_ready.csv
        - val_ready.csv
        - test_ready.csv

Expected outputs:
    processed/hierarchical/level3_family_submodels/<family>/
        - train.csv
        - val.csv
        - test.csv
        - label_mapping.pkl
        - label_mapping.json
        - summary.json
"""

import json
import pickle
from pathlib import Path

import pandas as pd

LABEL_COL = "label"

ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT.parent.parent
INPUT_DIR = EXPERIMENT_ROOT / "post_balancing_preprocessing_FINAL_base_balanced_only" / "exports"
OUTPUT_ROOT = EXPERIMENT_ROOT / "processed" / "hierarchical" / "level3_family_submodels"

TRAIN_PATH = INPUT_DIR / "train_ready.csv"
VAL_PATH = INPUT_DIR / "val_ready.csv"
TEST_PATH = INPUT_DIR / "test_ready.csv"

FAMILY_MAPPING = {
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


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[LABEL_COL] != "BENIGN"].copy()
    df["family"] = df[LABEL_COL].map(FAMILY_MAPPING)

    missing = sorted(df.loc[df["family"].isna(), LABEL_COL].unique().tolist())
    if missing:
        raise ValueError(f"Missing family mapping for labels: {missing}")

    return df


def build_family_split(df: pd.DataFrame, family: str, label_map: dict[str, int]) -> pd.DataFrame:
    family_df = df[df["family"] == family].copy()
    family_df["subtype_id"] = family_df[LABEL_COL].map(label_map).astype(int)
    return family_df


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        assert_exists(path)

    train_df = prepare_split(pd.read_csv(TRAIN_PATH))
    val_df = prepare_split(pd.read_csv(VAL_PATH))
    test_df = prepare_split(pd.read_csv(TEST_PATH))

    global_summary: dict[str, dict] = {}
    for family in sorted(set(FAMILY_MAPPING.values())):
        family_out = OUTPUT_ROOT / family
        family_out.mkdir(parents=True, exist_ok=True)

        train_family = train_df[train_df["family"] == family].copy()
        val_family = val_df[val_df["family"] == family].copy()
        test_family = test_df[test_df["family"] == family].copy()

        all_labels = sorted(
            set(train_family[LABEL_COL].unique().tolist())
            | set(val_family[LABEL_COL].unique().tolist())
            | set(test_family[LABEL_COL].unique().tolist())
        )

        if not all_labels:
            summary = {
                "family": family,
                "note": "no rows found in any split",
                "n_classes": 0,
                "labels": [],
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
            }
            save_json(summary, family_out / "summary.json")
            global_summary[family] = summary
            continue

        label_map = {label: idx for idx, label in enumerate(all_labels)}
        save_pickle(label_map, family_out / "label_mapping.pkl")
        save_json(label_map, family_out / "label_mapping.json")

        train_out = build_family_split(train_df, family, label_map)
        val_out = build_family_split(val_df, family, label_map)
        test_out = build_family_split(test_df, family, label_map)

        if len(train_out) > 0:
            train_out.to_csv(family_out / "train.csv", index=False)
        if len(val_out) > 0:
            val_out.to_csv(family_out / "val.csv", index=False)
        if len(test_out) > 0:
            test_out.to_csv(family_out / "test.csv", index=False)

        summary = {
            "family": family,
            "n_classes": len(all_labels),
            "labels": all_labels,
            "label_mapping": label_map,
            "n_train": int(len(train_out)),
            "n_val": int(len(val_out)),
            "n_test": int(len(test_out)),
            "train_distribution": train_out[LABEL_COL].value_counts().to_dict() if len(train_out) else {},
            "val_distribution": val_out[LABEL_COL].value_counts().to_dict() if len(val_out) else {},
            "test_distribution": test_out[LABEL_COL].value_counts().to_dict() if len(test_out) else {},
        }
        save_json(summary, family_out / "summary.json")
        global_summary[family] = summary

    save_json(global_summary, OUTPUT_ROOT / "global_summary.json")
    print("Level-3 subtype datasets created successfully.")
    print("Input:", INPUT_DIR)
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
