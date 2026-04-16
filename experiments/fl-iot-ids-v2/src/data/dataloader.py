from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split

from src.data.datasets.flat_dataset import IoTLocalDataset


def create_dataloaders_for_node(
    node_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/eval dataloaders for one federated client node.

    Expected:
        node_dir/
            train_preprocessed.npz

    Args:
        node_dir: Directory containing one node dataset
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        train_ratio: Fraction used for train, rest for eval
        seed: Random seed for deterministic split

    Returns:
        (train_loader, eval_loader)
    """
    node_dir = Path(node_dir)
    npz_path = node_dir / "train_preprocessed.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    dataset = IoTLocalDataset(npz_path)

    train_size = int(len(dataset) * train_ratio)
    eval_size = len(dataset) - train_size

    if train_size <= 0 or eval_size <= 0:
        raise ValueError(
            f"Invalid split sizes for {npz_path}: "
            f"train_size={train_size}, eval_size={eval_size}"
        )

    generator = None
    try:
        import torch

        generator = torch.Generator().manual_seed(seed)
    except Exception:
        generator = None

    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, eval_loader


def resolve_node_processed_dir(
    scenario: str,
    node_id: str,
    processed_root: str | Path = "data/processed",
) -> Path:
    """
    Resolve node processed directory from scenario + node_id.

    Example:
        data/processed/normal_noniid/node1/
    """
    processed_root = Path(processed_root)
    return processed_root / scenario / node_id


def create_dataloaders_for_scenario_node(
    scenario: str,
    node_id: str,
    processed_root: str | Path = "data/processed",
    batch_size: int = 256,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders using scenario-based structure.

    Expected:
        data/processed/<scenario>/<node_id>/train_preprocessed.npz
    """
    node_dir = resolve_node_processed_dir(
        scenario=scenario,
        node_id=node_id,
        processed_root=processed_root,
    )

    return create_dataloaders_for_node(
        node_dir=node_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        seed=seed,
    )