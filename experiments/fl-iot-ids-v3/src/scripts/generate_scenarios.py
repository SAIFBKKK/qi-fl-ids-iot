"""
generate_scenarios.py — partition + preprocess for named non-IID scenarios.

Usage:
    python -m src.scripts.generate_scenarios --scenario normal_noniid
    python -m src.scripts.generate_scenarios --scenario rare_expert
    python -m src.scripts.generate_scenarios --scenario absent_local

Each run:
  1. Partitions the global dataset according to the scenario rules.
  2. Saves raw CSVs  → data/raw/{scenario}/node{i}/train.csv
  3. Scales features with the pre-fitted global scaler (never refits).
  4. Saves NPZ files → data/processed/{scenario}/node{i}/train_preprocessed.npz
  5. Writes a manifest → data/splits/{scenario}_manifest.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DATASET_CSV,
    DATASET_PARQUET,
    get_processed_path,
    get_raw_path,
)
from src.common.utils import set_seed
from src.scripts.prepare_partitions import compute_node_stats, dirichlet_partition
from src.scripts.preprocess_node_data import load_global_scaler


logger = get_logger("generate_scenarios")

# ── constants ────────────────────────────────────────────────────────────────
LABEL_COL = "label_id"
BENIGN_CLASS = 1        # BenignTraffic — must be present in every node, every scenario
NUM_NODES = 3
DEFAULT_SEED = 42

SUPPORTED_SCENARIOS = ("normal_noniid", "rare_expert", "absent_local")

# rare_expert — domain-specific expert classes for node3 (see label_mapping.json)
EXPERT_CLASS_IDS: Set[int] = {
    # Very rare / critical application-layer attacks
    0,   # Backdoor_Malware
    2,   # BrowserHijacking
    3,   # CommandInjection
    17,  # DictionaryBruteForce
    30,  # SqlInjection
    31,  # Uploading_Attack
    33,  # XSS
    # Rare but exploitable
    11,  # DDoS-SlowLoris
    16,  # DNS_Spoofing
    18,  # DoS-HTTP_Flood
    22,  # MITM-ArpSpoofing
    26,  # Recon-HostDiscovery
    27,  # Recon-OSScan
    28,  # Recon-PingSweep
    29,  # Recon-PortScan
    32,  # VulnerabilityScan
}
NORMAL_CLASS_IDS: Set[int] = {c for c in range(34) if c not in EXPERT_CLASS_IDS}
# NORMAL_CLASS_IDS = {1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25}

NODE3_MAX_SAMPLES = 2_000_000      # hard cap on node3 (rare-expert only) rows

# absent_local
ABSENT_FRACTION = 0.30             # fraction of classes absent per node
ABSENT_ALPHA = 0.3                 # sharper Dirichlet heterogeneity
ABSENT_SEED_OFFSET = 100           # offset so absent-class RNG differs from partition RNG


# ── benign helpers ───────────────────────────────────────────────────────────

def _split_benign(df_benign: pd.DataFrame, num_nodes: int, seed: int) -> List[pd.DataFrame]:
    """Shuffle and split benign rows into num_nodes equal-ish portions."""
    shuffled = df_benign.sample(frac=1, random_state=seed).reset_index(drop=True)
    return [s.reset_index(drop=True) for s in np.array_split(shuffled, num_nodes)]


def _validate_benign_presence(parts: Dict[str, pd.DataFrame]) -> None:
    """Raise AssertionError immediately if any node is missing BenignTraffic."""
    for node_id, df in parts.items():
        benign_count = int((df[LABEL_COL] == BENIGN_CLASS).sum())
        if benign_count == 0:
            raise AssertionError(
                f"CRITICAL: {node_id} is missing BenignTraffic (class {BENIGN_CLASS}). "
                "IDS models cannot learn normal behaviour without benign samples."
            )
        logger.info(
            "Benign check ✓ %s → %d BenignTraffic samples (%.1f%% of node)",
            node_id, benign_count, 100.0 * benign_count / max(len(df), 1),
        )


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_dataset() -> pd.DataFrame:
    if DATASET_PARQUET.exists():
        logger.info("Loading Parquet: %s", DATASET_PARQUET)
        return pd.read_parquet(DATASET_PARQUET)
    if DATASET_CSV.exists():
        logger.info("Parquet not found, loading CSV: %s", DATASET_CSV)
        return pd.read_csv(DATASET_CSV)
    raise FileNotFoundError(
        f"Dataset not found. Tried:\n  {DATASET_PARQUET}\n  {DATASET_CSV}"
    )


# ── scenario partitioners ─────────────────────────────────────────────────────

def _partition_normal_noniid(
    df: pd.DataFrame, seed: int
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Dirichlet(alpha=0.5) partition across 3 nodes.

    BenignTraffic is extracted first and split evenly so every node is
    guaranteed a balanced view of normal traffic regardless of the Dirichlet
    draw.  Attack classes follow the standard non-IID Dirichlet distribution.
    """
    # Step 1 — Isolate benign traffic
    df_benign = df[df[LABEL_COL] == BENIGN_CLASS].copy()
    df_non_benign = df[df[LABEL_COL] != BENIGN_CLASS].copy()
    logger.info(
        "normal_noniid: benign=%d rows | non-benign=%d rows (%d classes)",
        len(df_benign), len(df_non_benign), df_non_benign[LABEL_COL].nunique(),
    )

    # Step 2 — Dirichlet partition on attack classes only (33 classes)
    parts_non_benign = dirichlet_partition(df_non_benign, NUM_NODES, alpha=0.5, seed=seed)

    # Step 3 — Split benign evenly, merge into each node
    benign_splits = _split_benign(df_benign, NUM_NODES, seed)
    parts: Dict[str, pd.DataFrame] = {}
    for i, node_id in enumerate(["node1", "node2", "node3"]):
        parts[node_id] = pd.concat(
            [parts_non_benign[node_id], benign_splits[i]], ignore_index=True
        )
        logger.info(
            "normal_noniid: %s | total=%d | benign=%d | attack=%d",
            node_id, len(parts[node_id]),
            len(benign_splits[i]), len(parts_non_benign[node_id]),
        )

    meta: dict = {"alpha": 0.5, "type": "dirichlet", "benign_handling": "evenly_split"}
    return parts, meta


def _partition_rare_expert(
    df: pd.DataFrame, seed: int
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Hard class-separation with guaranteed BenignTraffic in every node.

      node1, node2 — NORMAL_CLASS_IDS (17 attack classes) + BenignTraffic
                     Dirichlet(0.5) on attack classes; benign split evenly.
                     Hard filter applied after Dirichlet as a second defensive
                     layer against any future upstream changes.
      node3        — EXPERT_CLASS_IDS (16 attack classes) + BenignTraffic
                     ALL available expert rows (capped at NODE3_MAX_SAMPLES),
                     plus an equal share of benign traffic.

    Validation order:
      1. node1/node2 must be disjoint from EXPERT_CLASS_IDS
      2. node3 must be a subset of EXPERT_CLASS_IDS ∪ {BENIGN_CLASS}
      3. Every node must contain BENIGN_CLASS  (via _validate_benign_presence)
    """
    # Step 1 — Isolate benign; partition will only touch attack classes
    df_benign = df[df[LABEL_COL] == BENIGN_CLASS].copy()
    df_non_benign = df[df[LABEL_COL] != BENIGN_CLASS].copy()

    # Split attack classes by domain
    df_rare   = df_non_benign[df_non_benign[LABEL_COL].isin(EXPERT_CLASS_IDS)].copy()
    df_normal = df_non_benign[df_non_benign[LABEL_COL].isin(NORMAL_CLASS_IDS)].copy()
    logger.info(
        "rare_expert: benign=%d | df_rare=%d rows (%d classes) | df_normal=%d rows (%d classes)",
        len(df_benign),
        len(df_rare),   df_rare[LABEL_COL].nunique(),
        len(df_normal), df_normal[LABEL_COL].nunique(),
    )

    # Step 2 — Dirichlet(0.5) on normal attack classes → node1 and node2 only
    normal_parts = dirichlet_partition(df_normal, num_nodes=2, alpha=0.5, seed=seed)
    node1_df = normal_parts["node1"]
    node2_df = normal_parts["node2"]

    # Step 3 — Hard filter: remove any expert class that could have slipped through
    node1_df = node1_df[~node1_df[LABEL_COL].isin(EXPERT_CLASS_IDS)].reset_index(drop=True)
    node2_df = node2_df[~node2_df[LABEL_COL].isin(EXPERT_CLASS_IDS)].reset_index(drop=True)

    # Step 4 — node3 attack rows: all expert-class rows, capped at NODE3_MAX_SAMPLES
    # df_rare was already built with isin(EXPERT_CLASS_IDS); we apply a second
    # strict filter here so the guarantee holds regardless of upstream changes.
    df_rare = df_rare[df_rare[LABEL_COL].isin(EXPERT_CLASS_IDS)].reset_index(drop=True)
    if len(df_rare) > NODE3_MAX_SAMPLES:
        node3_attack = df_rare.sample(n=NODE3_MAX_SAMPLES, random_state=seed).reset_index(drop=True)
    else:
        node3_attack = df_rare.reset_index(drop=True)
    # Strict post-filter: enforce exactly EXPERT_CLASS_IDS — catches any row that
    # could slip through if the cap/sample step ever changes.
    node3_attack = node3_attack[node3_attack[LABEL_COL].isin(EXPERT_CLASS_IDS)].reset_index(drop=True)
    logger.info(
        "rare_expert: node3_attack strict filter → %d rows | %d distinct expert classes: %s",
        len(node3_attack),
        node3_attack[LABEL_COL].nunique(),
        sorted(node3_attack[LABEL_COL].unique()),
    )

    # Step 5 — Distribute benign evenly to all three nodes
    benign_splits = _split_benign(df_benign, NUM_NODES, seed)
    node1_df = pd.concat([node1_df, benign_splits[0]], ignore_index=True)
    node2_df = pd.concat([node2_df, benign_splits[1]], ignore_index=True)
    node3_df = pd.concat([node3_attack, benign_splits[2]], ignore_index=True)

    # Step 6 — Mandatory separation validation
    node1_cls = set(node1_df[LABEL_COL].unique())
    node2_cls = set(node2_df[LABEL_COL].unique())
    node3_cls = set(node3_df[LABEL_COL].unique())

    if not node1_cls.isdisjoint(EXPERT_CLASS_IDS):
        leaked = sorted(node1_cls & EXPERT_CLASS_IDS)
        raise AssertionError(f"LEAKAGE: node1 contains expert classes {leaked}")
    if not node2_cls.isdisjoint(EXPERT_CLASS_IDS):
        leaked = sorted(node2_cls & EXPERT_CLASS_IDS)
        raise AssertionError(f"LEAKAGE: node2 contains expert classes {leaked}")
    # node3 is allowed to contain BENIGN_CLASS in addition to expert classes
    allowed_node3 = EXPERT_CLASS_IDS | {BENIGN_CLASS}
    if not node3_cls.issubset(allowed_node3):
        extra = sorted(node3_cls - allowed_node3)
        raise AssertionError(f"LEAKAGE: node3 contains unexpected classes {extra}")

    # Step 7 — Per-node class logs
    logger.info("rare_expert: node1 classes = %s", sorted(node1_cls))
    logger.info("rare_expert: node2 classes = %s", sorted(node2_cls))
    logger.info("rare_expert: node3 classes = %s", sorted(node3_cls))
    logger.info(
        "rare_expert: node1=%d rows (attack=%d benign=%d) | node2=%d rows (attack=%d benign=%d) | node3=%d rows (attack=%d benign=%d)",
        len(node1_df), len(normal_parts["node1"]), len(benign_splits[0]),
        len(node2_df), len(normal_parts["node2"]), len(benign_splits[1]),
        len(node3_df), len(node3_attack), len(benign_splits[2]),
    )
    logger.info("rare_expert: separation validated — no expert-class leakage in node1/node2")

    parts = {"node1": node1_df, "node2": node2_df, "node3": node3_df}

    node_expert_presence = {
        "node1": sorted(node1_cls & EXPERT_CLASS_IDS),   # always []
        "node2": sorted(node2_cls & EXPERT_CLASS_IDS),   # always []
        "node3": sorted(node3_cls & EXPERT_CLASS_IDS),
    }

    meta: dict = {
        "alpha": 0.5,
        "type": "rare_expert",
        "benign_handling": "evenly_split",
        "expert_class_ids": sorted(EXPERT_CLASS_IDS),
        "normal_class_ids": sorted(NORMAL_CLASS_IDS),
        "node3_max_samples": NODE3_MAX_SAMPLES,
        "expert_classes_present_per_node": node_expert_presence,
    }
    return parts, meta


def _partition_absent_local(
    df: pd.DataFrame, seed: int
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Each node is missing ABSENT_FRACTION of attack classes (different subset per node).
    Uses Dirichlet(0.3) for stronger heterogeneity.

    BenignTraffic (class 1) is extracted before partitioning and redistributed
    evenly — it is NEVER eligible for removal from any node.

    Coverage over attack classes is guaranteed by construction: a class is only
    marked absent from a node when at least one other node actually holds samples
    of that class after the Dirichlet partition.
    """
    # Step 1 — Isolate benign; Dirichlet and absent-selection run on attack classes only
    df_benign = df[df[LABEL_COL] == BENIGN_CLASS].copy()
    df_non_benign = df[df[LABEL_COL] != BENIGN_CLASS].copy()
    logger.info(
        "absent_local: benign=%d rows | attack=%d rows (%d classes)",
        len(df_benign), len(df_non_benign), df_non_benign[LABEL_COL].nunique(),
    )

    # Step 2 — Dirichlet(0.3) base partition on attack classes only
    base = dirichlet_partition(df_non_benign, NUM_NODES, alpha=ABSENT_ALPHA, seed=seed)

    # Step 3 — Record which attack classes each node received from Dirichlet
    node_present: Dict[int, Set[int]] = {
        i: set(int(c) for c in base[f"node{i + 1}"][LABEL_COL].unique())
        for i in range(NUM_NODES)
    }

    num_classes = len(df_non_benign[LABEL_COL].unique())
    num_absent = max(1, int(num_classes * ABSENT_FRACTION))
    rng = np.random.default_rng(seed + ABSENT_SEED_OFFSET)

    # Step 4 — Randomly assign absent classes for each node.
    # BENIGN_CLASS is never in node_present (we partitioned df_non_benign), so it
    # cannot be selected here — but we filter it out explicitly as a safety guard.
    absent_per_node: Dict[int, Set[int]] = {i: set() for i in range(NUM_NODES)}
    for i in range(NUM_NODES):
        candidates = sorted(node_present[i] - {BENIGN_CLASS})  # defensive guard
        n_remove = min(num_absent, len(candidates))
        if n_remove > 0:
            chosen = rng.choice(candidates, size=n_remove, replace=False)
            absent_per_node[i] = set(int(c) for c in chosen)

    # Post-hoc coverage fix: reinstate any attack class that would vanish globally
    all_attack_classes = set(int(c) for c in df_non_benign[LABEL_COL].unique())
    fixed: List[int] = []
    for cls in sorted(all_attack_classes):
        will_survive = [
            i for i in range(NUM_NODES)
            if cls in node_present[i] and cls not in absent_per_node[i]
        ]
        if not will_survive:
            nodes_with_cls = [i for i in range(NUM_NODES) if cls in node_present[i]]
            if nodes_with_cls:
                best = max(
                    nodes_with_cls,
                    key=lambda i: int((base[f"node{i + 1}"][LABEL_COL] == cls).sum()),
                )
                absent_per_node[best].discard(cls)
                fixed.append(cls)
    if fixed:
        logger.warning(
            "absent_local: %d attack class(es) reinstated to preserve global coverage: %s",
            len(fixed), fixed,
        )

    # Step 5 — Filter each node, then add benign
    benign_splits = _split_benign(df_benign, NUM_NODES, seed)
    parts: Dict[str, pd.DataFrame] = {}
    absent_info: Dict[str, List[int]] = {}
    for i in range(NUM_NODES):
        node_id = f"node{i + 1}"
        mask = ~base[node_id][LABEL_COL].isin(absent_per_node[i])
        attack_part = base[node_id][mask].reset_index(drop=True)
        parts[node_id] = pd.concat([attack_part, benign_splits[i]], ignore_index=True)
        absent_info[node_id] = sorted(absent_per_node[i])
        logger.info(
            "absent_local: %s | total=%d (attack=%d benign=%d) | absent_classes=%s",
            node_id, len(parts[node_id]),
            len(attack_part), len(benign_splits[i]),
            absent_info[node_id],
        )

    meta: dict = {
        "alpha": ABSENT_ALPHA,
        "type": "absent_local",
        "benign_handling": "evenly_split_never_absent",
        "absent_fraction": ABSENT_FRACTION,
        "absent_classes_per_node": absent_info,
    }
    return parts, meta


# ── preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_and_save(
    part_df: pd.DataFrame,
    node_id: str,
    scenario: str,
    scaler,
    scaler_name: str,
) -> None:
    """Scale features with the global scaler and write a compressed NPZ."""
    feature_cols = [c for c in part_df.columns if c != LABEL_COL]
    y = part_df[LABEL_COL].to_numpy(dtype=np.int64)
    X_raw = part_df[feature_cols].to_numpy(dtype=np.float64)

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    node_mean = float(X_scaled.mean())
    node_std = float(X_scaled.std())
    logger.info(
        "%s/%s | scaler=%s | mean=%.4f  std=%.4f  (expected ≈0, ≈1)",
        scenario, node_id, scaler_name, node_mean, node_std,
    )
    if abs(node_mean) > 0.1:
        logger.warning("%s/%s: post-scale mean=%.4f far from 0", scenario, node_id, node_mean)
    if abs(node_std - 1.0) > 0.5:
        logger.warning(
            "%s/%s: std=%.4f — strong distribution shift (expected in absent_local)",
            scenario, node_id, node_std,
        )
    elif abs(node_std - 1.0) > 0.2:
        logger.info(
            "%s/%s: std=%.4f — mild distribution shift (normal for non-IID)",
            scenario, node_id, node_std,
        )

    out_path = get_processed_path(scenario, node_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_scaled,
        y=y,
        feature_names=np.array(feature_cols, dtype=object),
    )
    logger.info("Saved NPZ -> %s | X=%s | y=%s", out_path, X_scaled.shape, y.shape)


# ── manifest ──────────────────────────────────────────────────────────────────

def _save_manifest(scenario: str, parts: Dict[str, pd.DataFrame], meta: dict) -> None:
    source_path = DATASET_PARQUET if DATASET_PARQUET.exists() else DATASET_CSV
    manifest: dict = {
        "scenario": scenario,
        "dataset_source": str(source_path),
        "label_column": LABEL_COL,
        "num_nodes": NUM_NODES,
        **meta,
        "nodes": {},
    }
    for node_id, part_df in parts.items():
        raw_path = get_raw_path(scenario, node_id)
        top5 = (
            part_df[LABEL_COL]
            .value_counts()
            .head(5)
            .to_dict()
        )
        manifest["nodes"][node_id] = {
            "raw_csv": str(raw_path),
            **compute_node_stats(part_df),
            "top5_classes": {int(k): int(v) for k, v in top5.items()},
        }

    manifest_dir = DATA_DIR / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{scenario}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest -> %s", manifest_path)


# ── main ──────────────────────────────────────────────────────────────────────

def run_scenario(scenario: str, seed: int = DEFAULT_SEED) -> None:
    set_seed(seed)
    logger.info("=== Generating scenario: %s (seed=%d) ===", scenario, seed)

    # 1. Load global scaler (must exist — do NOT refit)
    scaler, scaler_name = load_global_scaler(ARTIFACTS_DIR)
    logger.info("Global scaler loaded: %s", scaler_name)

    # 2. Load full dataset
    df = _load_dataset()
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COL}' not found. Available: {list(df.columns)}"
        )
    logger.info("Dataset loaded: shape=%s", df.shape)

    # 3. Partition
    if scenario == "normal_noniid":
        parts, meta = _partition_normal_noniid(df, seed)
    elif scenario == "rare_expert":
        parts, meta = _partition_rare_expert(df, seed)
    elif scenario == "absent_local":
        parts, meta = _partition_absent_local(df, seed)
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {SUPPORTED_SCENARIOS}")

    # 3b. Hard guarantee: every node must contain BenignTraffic before any data touches disk
    _validate_benign_presence(parts)

    # 4. Save raw CSVs + preprocess + save NPZs
    for node_id, part_df in parts.items():
        raw_path = get_raw_path(scenario, node_id)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        part_df.to_csv(raw_path, index=False)
        logger.info("Raw CSV -> %s (%d rows)", raw_path, len(part_df))

        _preprocess_and_save(part_df, node_id, scenario, scaler, scaler_name)

    # 5. Manifest
    _save_manifest(scenario, parts, meta)

    # 6. Summary
    logger.info("=== Summary: %s ===", scenario)
    for node_id, part_df in parts.items():
        stats = compute_node_stats(part_df)
        top5 = part_df[LABEL_COL].value_counts().head(5)
        top5_str = "  ".join(f"cls{int(c)}={int(n)}" for c, n in top5.items())
        logger.info(
            "  %s | rows=%d | classes=%d | imbalance=%.1fx | rare=%d | top5: %s",
            node_id,
            stats["rows"],
            len(stats["class_distribution"]),
            stats["imbalance_ratio"],
            len(stats["rare_classes"]),
            top5_str,
        )
    logger.info("Done: %s", scenario)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and preprocess a named non-IID FL scenario."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=SUPPORTED_SCENARIOS,
        help="Scenario to generate.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    run_scenario(scenario=args.scenario, seed=args.seed)


if __name__ == "__main__":
    main()
