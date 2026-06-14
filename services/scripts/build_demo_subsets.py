"""
build_demo_subsets.py - Genere 6 sous-ensembles thematiques de CIC-IoT-2023
pour la demo microservices.

Usage: python services/scripts/build_demo_subsets.py

Genere dans data/cic-iot-2023/demo_subsets/:
- normal_traffic.parquet  (100% benign, 10K flows)
- ddos_burst.parquet      (10% benign + 90% DDoS-*, 10K flows)
- recon_scan.parquet      (80% benign + 20% Recon-*, 10K flows)
- mirai_wave.parquet      (60% benign + 40% Mirai-*, 10K flows)
- dos_slow.parquet        (50% benign + 50% DoS-*, 10K flows)
- mixed_chaos.parquet     (stratifie 34 classes, 10K flows)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PARQUET = (
    REPO_ROOT
    / "data"
    / "balancing_v3_fixed300k_outputs"
    / "balancing_v3_fixed300k_balanced.parquet"
)
BUNDLE_PATH = (
    REPO_ROOT
    / "experiments"
    / "fl-iot-ids-v3"
    / "outputs"
    / "deployment"
    / "baseline_fedavg_normal_classweights"
)
OUTPUT_DIR = REPO_ROOT / "data" / "cic-iot-2023" / "demo_subsets"
SEED = 42
TOTAL_FLOWS_PER_SUBSET = 10_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_demo_subsets")


def load_id_to_label() -> dict[int, str]:
    with open(BUNDLE_PATH / "label_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    if "id_to_label" in mapping:
        return {int(k): v for k, v in mapping["id_to_label"].items()}
    if "label_to_id" in mapping:
        return {int(v): k for k, v in mapping["label_to_id"].items()}
    return {int(k): v for k, v in mapping.items()}


ID_TO_LABEL = load_id_to_label()
BENIGN_IDS = [k for k, v in ID_TO_LABEL.items() if "Benign" in v or "BENIGN" in v.upper()]
DDOS_IDS = [k for k, v in ID_TO_LABEL.items() if "DDoS" in v]
RECON_IDS = [k for k, v in ID_TO_LABEL.items() if "Recon" in v]
MIRAI_IDS = [k for k, v in ID_TO_LABEL.items() if "Mirai" in v]
DOS_IDS = [k for k, v in ID_TO_LABEL.items() if "DoS" in v and "DDoS" not in v]


def require_ids(name: str, ids: list[int]) -> None:
    if not ids:
        raise ValueError(f"No label IDs found for category {name}")


def build_mix(
    df: pd.DataFrame,
    mix: dict[str, tuple[list[int], float]],
    total: int = TOTAL_FLOWS_PER_SUBSET,
) -> pd.DataFrame:
    parts = []
    allocated = 0
    items = list(mix.items())

    for index, (name, (ids, ratio)) in enumerate(items):
        require_ids(name, ids)
        n = total - allocated if index == len(items) - 1 else int(total * ratio)
        allocated += n

        subset = df[df["label_id"].isin(ids)]
        if len(subset) < n:
            logger.warning("%s: only %s flows available, requested %s", name, len(subset), n)
            sampled = subset.sample(n=n, replace=True, random_state=SEED)
        else:
            sampled = subset.sample(n=n, random_state=SEED)
        parts.append(sampled)

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def build_mixed_chaos(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(df["label_id"].unique())
    base = TOTAL_FLOWS_PER_SUBSET // len(classes)
    remainder = TOTAL_FLOWS_PER_SUBSET % len(classes)
    parts = []

    for index, label_id in enumerate(classes):
        n = base + (1 if index < remainder else 0)
        subset = df[df["label_id"] == label_id]
        replace = len(subset) < n
        parts.append(subset.sample(n=n, replace=replace, random_state=SEED))

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def write_subset(name: str, subset_df: pd.DataFrame) -> None:
    path = OUTPUT_DIR / f"{name}.parquet"
    subset_df.to_parquet(path, compression="snappy", index=False)
    logger.info(
        "%s: %s flows, %s classes, size=%.0f KB",
        path.name,
        len(subset_df),
        subset_df["label_id"].nunique(),
        path.stat().st_size / 1024,
    )


def main() -> None:
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(f"Source parquet not found: {SOURCE_PARQUET}")
    if not (BUNDLE_PATH / "label_mapping.json").exists():
        raise FileNotFoundError(f"label_mapping.json not found in: {BUNDLE_PATH}")

    logger.info(
        "Found classes: BENIGN=%s, DDoS=%s, Recon=%s, Mirai=%s, DoS=%s",
        len(BENIGN_IDS),
        len(DDOS_IDS),
        len(RECON_IDS),
        len(MIRAI_IDS),
        len(DOS_IDS),
    )
    logger.info("Loading source: %s", SOURCE_PARQUET)
    df = pd.read_parquet(SOURCE_PARQUET)
    logger.info("Source loaded: %s flows, %s classes", len(df), df["label_id"].nunique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subsets = {
        "normal_traffic": {"benign": (BENIGN_IDS, 1.0)},
        "ddos_burst": {
            "benign": (BENIGN_IDS, 0.10),
            "ddos": (DDOS_IDS, 0.90),
        },
        "recon_scan": {
            "benign": (BENIGN_IDS, 0.80),
            "recon": (RECON_IDS, 0.20),
        },
        "mirai_wave": {
            "benign": (BENIGN_IDS, 0.60),
            "mirai": (MIRAI_IDS, 0.40),
        },
        "dos_slow": {
            "benign": (BENIGN_IDS, 0.50),
            "dos": (DOS_IDS, 0.50),
        },
    }

    for name, mix in subsets.items():
        logger.info("Building %s...", name)
        write_subset(name, build_mix(df, mix))

    logger.info("Building mixed_chaos...")
    write_subset("mixed_chaos", build_mixed_chaos(df))

    total_size_mb = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.parquet")) / 1024 / 1024
    logger.info("Total demo subsets size: %.1f MB", total_size_mb)
    logger.info("Files created in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
