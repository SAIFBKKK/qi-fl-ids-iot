from pathlib import Path
from torch.utils.data import DataLoader, random_split

from src.data.dataset import IoTLocalDataset


def create_dataloaders_for_node(
    node_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
    train_ratio: float = 0.8,
):
    """Create train and eval dataloaders for a federated client node.
    
    Args:
        node_dir: Directory containing node data (with train_preprocessed.npz)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_ratio: Ratio of data to use for training (rest goes to eval)
    
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    node_dir = Path(node_dir)
    npz_path = node_dir / "train_preprocessed.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")
    
    dataset = IoTLocalDataset(npz_path)
    
    # Split dataset into train and eval
    train_size = int(len(dataset) * train_ratio)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
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