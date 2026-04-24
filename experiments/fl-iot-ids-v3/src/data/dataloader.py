from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import IoTLocalDataset


def create_dataloaders_for_node(
    node_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
):
    """Create train and eval dataloaders for a federated client node.
    
    Args:
        node_dir: Directory containing node data (with train_preprocessed.npz)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, eval_loader)
    """
    node_dir = Path(node_dir)
    train_npz = node_dir / "train_preprocessed.npz"
    val_npz = node_dir / "val_preprocessed.npz"

    missing = [str(path) for path in (train_npz, val_npz) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Explicit train/val NPZ splits are required; random post-preprocessing "
            f"splits are forbidden to avoid leakage. Missing: {missing}"
        )

    train_dataset = IoTLocalDataset(train_npz)
    eval_dataset = IoTLocalDataset(val_npz)

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
