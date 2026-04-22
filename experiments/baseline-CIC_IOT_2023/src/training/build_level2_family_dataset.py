"""
Build the Level-2 hierarchical dataset from preprocessing exports.

This utility was extracted from `baseline-CIC_IOT_2023_v2` during cleanup so
the kept `baseline-CIC_IOT_2023` experiment retains the useful dataset-builder
logic before the donor folder is retired.

Expected input files under the experiment root:
    post_balancing_preprocessing_FINAL_base_balanced_only/exports/
        - train_ready.csv
        - val_ready.csv
        - test_ready.csv

Expected outputs:
    processed/hierarchical/level2_family/
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
OUTPUT_DIR = EXPERIMENT_ROOT / "processed" / "hierarchical" / "level2_family"

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


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def prepare_split(df: pd.DataFrame, family_to_id: dict[str, int]) -> pd.DataFrame:
    df = df[df[LABEL_COL] != "BENIGN"].copy()
    df["family"] = df[LABEL_COL].map(FAMILY_MAPPING)

    missing = sorted(df.loc[df["family"].isna(), LABEL_COL].unique().tolist())
    if missing:
        raise ValueError(f"Missing family mapping for labels: {missing}")

    df["family_id"] = df["family"].map(family_to_id).astype(int)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        assert_exists(path)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    family_names = sorted(set(FAMILY_MAPPING.values()))
    family_to_id = {name: idx for idx, name in enumerate(family_names)}

    train_out = prepare_split(train_df, family_to_id)
    val_out = prepare_split(val_df, family_to_id)
    test_out = prepare_split(test_df, family_to_id)

    train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_out.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)

    save_pickle(family_to_id, OUTPUT_DIR / "label_mapping.pkl")
    save_json(family_to_id, OUTPUT_DIR / "label_mapping.json")

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "n_train": int(len(train_out)),
        "n_val": int(len(val_out)),
        "n_test": int(len(test_out)),
        "n_families": int(len(family_to_id)),
        "families": family_to_id,
        "train_distribution": train_out["family"].value_counts().to_dict(),
        "val_distribution": val_out["family"].value_counts().to_dict(),
        "test_distribution": test_out["family"].value_counts().to_dict(),
    }
    save_json(summary, OUTPUT_DIR / "summary.json")

    print("Level-2 family dataset created successfully.")
    print("Input:", INPUT_DIR)
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
