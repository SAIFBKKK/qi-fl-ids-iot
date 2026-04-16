from __future__ import annotations

from src.data.datasets.flat_dataset import FlatTensorDataset, load_npz_xy


class HierarchicalDataset(FlatTensorDataset):
    """
    Temporary thin wrapper.
    Later this can expose:
    - level1 labels (benign vs attack)
    - level2 family labels
    - level3 fine-grained labels
    """
    pass


def load_hierarchical_npz(path: str):
    return load_npz_xy(path)