from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.paths import DATA_DIR


logger = get_logger("prepare_partitions")


DEFAULT_SEED = 42
ROW_ID_COL = "__row_id"
DEFAULT_SOURCE_PARQUET = Path(
    r"E:\dataset\CICIoT2023\balancing_v3_fixed300k_outputs\balancing_v3_fixed300k_balanced.parquet"
)

NODE_IDS = ["node1", "node2", "node3"]
DIRICHLET_ALPHA = 0.5
RARE_EXPERT_NODE = "node3"

# Tu peux ajuster cette liste plus tard si besoin
RARE_CLASSES = {
    0,   # Backdoor_Malware
    3,   # CommandInjection
    30,  # SqlInjection
    31,  # Uploading_Attack
    33,  # XSS
}

# Dans absent_local, on retire seulement des classes "communes"
# jamais les classes rares
COMMON_CLASSES_TO_DROP_BY_NODE = {
    "node1": {5, 10, 14},   # exemples classes communes/frequentes
    "node2": {6, 11, 18},
    "node3": {13, 19, 21},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_label_column(df: pd.DataFrame) -> str:
    for col in ["label_id", "label", "Label"]:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found in dataframe columns={list(df.columns)}")


def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if ROW_ID_COL in df.columns:
        if df[ROW_ID_COL].duplicated().any():
            raise ValueError(f"Column {ROW_ID_COL!r} must be unique.")
        return df.copy()
    out = df.copy()
    out.insert(0, ROW_ID_COL, np.arange(len(out), dtype=np.int64))
    return out


def validate_disjoint_partitions(partitions: Dict[str, pd.DataFrame]) -> Dict:
    """Raise if any source row is assigned to more than one client."""
    owners: dict[int, str] = {}
    intersections: list[dict[str, object]] = []
    for node_id, part_df in partitions.items():
        if ROW_ID_COL not in part_df.columns:
            raise ValueError(f"{node_id} partition is missing {ROW_ID_COL}")
        for row_id in part_df[ROW_ID_COL].astype(int).tolist():
            previous = owners.get(row_id)
            if previous is not None:
                intersections.append(
                    {"row_id": row_id, "first_node": previous, "second_node": node_id}
                )
            else:
                owners[row_id] = node_id

    if intersections:
        raise AssertionError(
            "Inter-client leakage detected: duplicated row ids across partitions. "
            f"First duplicates: {intersections[:10]}"
        )

    return {
        "row_id_column": ROW_ID_COL,
        "disjoint": True,
        "assigned_unique_rows": len(owners),
    }


def save_node_csv(df: pd.DataFrame, scenario: str, node_id: str) -> Path:
    out_dir = DATA_DIR / "raw" / scenario / node_id
    ensure_dir(out_dir)
    out_path = out_dir / "train.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s rows -> %s", len(df), out_path)
    return out_path


def class_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    counts = df[label_col].value_counts().sort_index()
    return {str(int(k)) if str(k).isdigit() else str(k): int(v) for k, v in counts.items()}


def build_manifest(
    scenario: str,
    source_path: Path,
    full_df: pd.DataFrame,
    label_col: str,
    partitions: Dict[str, pd.DataFrame],
    extra: Dict | None = None,
) -> Dict:
    manifest = {
        "scenario": scenario,
        "source_path": str(source_path),
        "row_count_source": int(len(full_df)),
        "column_count_source": int(full_df.shape[1]),
        "label_column": label_col,
        "row_id_column": ROW_ID_COL,
        "nodes": {},
    }

    for node_id, part_df in partitions.items():
        manifest["nodes"][node_id] = {
            "rows": int(len(part_df)),
            "columns": int(part_df.shape[1]),
            "class_distribution": class_counts(part_df, label_col),
            "output_csv": str(DATA_DIR / "raw" / scenario / node_id / "train.csv"),
        }

    disjoint_proof = validate_disjoint_partitions(partitions)
    manifest["partition_disjointness"] = disjoint_proof

    if extra:
        manifest["extra"] = extra

    return manifest


def save_manifest(scenario: str, manifest: Dict) -> Path:
    out_dir = DATA_DIR / "raw" / scenario
    ensure_dir(out_dir)
    out_path = out_dir / "manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved manifest -> %s", out_path)
    return out_path


def sample_dirichlet_partition(
    df: pd.DataFrame,
    label_col: str,
    node_ids: List[str],
    alpha: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    node_chunks: Dict[str, List[pd.DataFrame]] = {node_id: [] for node_id in node_ids}

    for label_value, group in df.groupby(label_col):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)

        proportions = rng.dirichlet([alpha] * len(node_ids))
        counts = np.floor(proportions * n).astype(int)

        # corriger l'arrondi
        diff = n - counts.sum()
        for i in range(diff):
            counts[i % len(counts)] += 1

        start = 0
        for node_id, cnt in zip(node_ids, counts):
            if cnt > 0:
                node_chunks[node_id].append(group.iloc[start:start + cnt].copy())
            start += cnt

    partitions = {}
    for node_id in node_ids:
        if node_chunks[node_id]:
            partitions[node_id] = pd.concat(node_chunks[node_id], axis=0).sample(
                frac=1.0, random_state=seed
            ).reset_index(drop=True)
        else:
            partitions[node_id] = df.iloc[0:0].copy()

    return partitions


def build_normal_noniid(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    logger.info("Building scenario: normal_noniid")
    return sample_dirichlet_partition(
        df=df,
        label_col=label_col,
        node_ids=NODE_IDS,
        alpha=DIRICHLET_ALPHA,
        seed=seed,
    )


def build_absent_local(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """
    Règle validée:
    - toutes les classes rares restent présentes chez tous les nœuds
    - certaines classes communes manquent selon le nœud
    """
    logger.info("Building scenario: absent_local")

    base = sample_dirichlet_partition(
        df=df,
        label_col=label_col,
        node_ids=NODE_IDS,
        alpha=DIRICHLET_ALPHA,
        seed=seed,
    )

    # 1) retirer certaines classes communes sur certains nœuds
    removed_pool: List[pd.DataFrame] = []
    for node_id in NODE_IDS:
        part = base[node_id]
        to_drop = COMMON_CLASSES_TO_DROP_BY_NODE.get(node_id, set())

        mask_drop = part[label_col].isin(to_drop) & (~part[label_col].isin(RARE_CLASSES))
        removed = part.loc[mask_drop].copy()
        kept = part.loc[~mask_drop].copy()

        removed_pool.append(removed)
        base[node_id] = kept.reset_index(drop=True)

    removed_all = pd.concat(removed_pool, axis=0).reset_index(drop=True) if removed_pool else df.iloc[0:0].copy()

    # 2) redistribuer les lignes retirées seulement vers les autres nœuds autorisés
    redistribution_chunks: Dict[str, List[pd.DataFrame]] = {node_id: [] for node_id in NODE_IDS}
    if len(removed_all) > 0:
        for cls, group in removed_all.groupby(label_col):
            allowed_nodes = [
                node_id for node_id in NODE_IDS
                if cls not in COMMON_CLASSES_TO_DROP_BY_NODE.get(node_id, set())
            ]
            if not allowed_nodes:
                allowed_nodes = NODE_IDS

            parts = np.array_split(
                group.sample(frac=1.0, random_state=seed),
                len(allowed_nodes),
            )
            for node_id, chunk in zip(allowed_nodes, parts):
                if len(chunk) > 0:
                    redistribution_chunks[node_id].append(chunk.copy())

    for node_id in NODE_IDS:
        if redistribution_chunks[node_id]:
            base[node_id] = pd.concat(
                [base[node_id]] + redistribution_chunks[node_id],
                axis=0,
            ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 3) garantir que toutes les classes rares existent chez tous les nœuds
    # without duplicating source rows: move one row from the richest donor.
    rare_df = df[df[label_col].isin(RARE_CLASSES)].copy()
    for rare_cls in sorted(RARE_CLASSES):
        rare_group = rare_df[rare_df[label_col] == rare_cls]
        if rare_group.empty:
            continue

        for node_id in NODE_IDS:
            has_rare = (base[node_id][label_col] == rare_cls).any()
            if not has_rare:
                donor_candidates = [
                    candidate
                    for candidate in NODE_IDS
                    if int((base[candidate][label_col] == rare_cls).sum()) > 1
                ]
                if not donor_candidates:
                    logger.warning(
                        "Cannot move rare class %s to %s without duplicating rows",
                        rare_cls,
                        node_id,
                    )
                    continue
                donor = max(
                    donor_candidates,
                    key=lambda candidate: int((base[candidate][label_col] == rare_cls).sum()),
                )
                donor_mask = base[donor][label_col] == rare_cls
                row_to_move = base[donor].loc[donor_mask].iloc[[0]].copy()
                base[donor] = base[donor].drop(row_to_move.index).reset_index(drop=True)
                base[node_id] = pd.concat(
                    [base[node_id], row_to_move],
                    axis=0,
                ).reset_index(drop=True)

    for node_id in NODE_IDS:
        base[node_id] = base[node_id].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return base


def build_rare_expert(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    logger.info("Building scenario: rare_expert")

    base = sample_dirichlet_partition(
        df=df,
        label_col=label_col,
        node_ids=NODE_IDS,
        alpha=DIRICHLET_ALPHA,
        seed=seed,
    )

    rare_df = df[df[label_col].isin(RARE_CLASSES)].copy()

    if len(rare_df) == 0:
        logger.warning("No rare-class rows found, rare_expert will fallback to normal_noniid.")
        return base

    # Move rare rows exclusively to the expert node.  This enriches node3
    # without duplicating samples across clients.
    rare_chunks: list[pd.DataFrame] = []
    for node_id in NODE_IDS:
        rare_mask = base[node_id][label_col].isin(RARE_CLASSES)
        rare_chunks.append(base[node_id].loc[rare_mask].copy())
        if node_id != RARE_EXPERT_NODE:
            base[node_id] = base[node_id].loc[~rare_mask].reset_index(drop=True)

    all_rare = pd.concat(rare_chunks, axis=0).drop_duplicates(ROW_ID_COL)
    expert_non_rare = base[RARE_EXPERT_NODE].loc[
        ~base[RARE_EXPERT_NODE][label_col].isin(RARE_CLASSES)
    ].copy()
    base[RARE_EXPERT_NODE] = pd.concat(
        [expert_non_rare, all_rare],
        axis=0,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    validate_disjoint_partitions(base)

    return base


def save_scenario(
    scenario: str,
    source_path: Path,
    full_df: pd.DataFrame,
    label_col: str,
    partitions: Dict[str, pd.DataFrame],
    extra: Dict | None = None,
) -> None:
    for node_id, part_df in partitions.items():
        save_node_csv(part_df, scenario, node_id)

    manifest = build_manifest(
        scenario=scenario,
        source_path=source_path,
        full_df=full_df,
        label_col=label_col,
        partitions=partitions,
        extra=extra,
    )
    save_manifest(scenario, manifest)


def main() -> None:
    seed = DEFAULT_SEED
    source_path = DEFAULT_SOURCE_PARQUET

    if not source_path.exists():
        raise FileNotFoundError(f"Source parquet not found: {source_path}")

    logger.info("Loading source dataset: %s", source_path)
    df = ensure_row_id(pd.read_parquet(source_path))
    logger.info("Loaded dataframe shape=%s", df.shape)

    label_col = detect_label_column(df)
    logger.info("Detected label column: %s", label_col)

    ensure_dir(DATA_DIR / "raw")

    # normal_noniid
    normal_parts = build_normal_noniid(df, label_col, seed)
    save_scenario(
        scenario="normal_noniid",
        source_path=source_path,
        full_df=df,
        label_col=label_col,
        partitions=normal_parts,
        extra={
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "node_ids": NODE_IDS,
        },
    )

    # absent_local
    absent_parts = build_absent_local(df, label_col, seed)
    save_scenario(
        scenario="absent_local",
        source_path=source_path,
        full_df=df,
        label_col=label_col,
        partitions=absent_parts,
        extra={
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "node_ids": NODE_IDS,
            "rare_classes": sorted(RARE_CLASSES),
            "common_classes_to_drop_by_node": {
                k: sorted(list(v)) for k, v in COMMON_CLASSES_TO_DROP_BY_NODE.items()
            },
        },
    )

    # rare_expert
    rare_parts = build_rare_expert(df, label_col, seed)
    save_scenario(
        scenario="rare_expert",
        source_path=source_path,
        full_df=df,
        label_col=label_col,
        partitions=rare_parts,
        extra={
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "node_ids": NODE_IDS,
            "rare_classes": sorted(RARE_CLASSES),
            "expert_node": RARE_EXPERT_NODE,
        },
    )

    logger.info("All scenarios generated successfully.")


if __name__ == "__main__":
    main()
