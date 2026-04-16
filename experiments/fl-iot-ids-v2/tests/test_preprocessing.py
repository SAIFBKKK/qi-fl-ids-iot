import numpy as np

from src.data.datasets.flat_dataset import FlatTensorDataset


def test_flat_tensor_dataset_shapes():
    x = np.random.randn(10, 33).astype(np.float32)
    y = np.random.randint(0, 34, size=(10,), dtype=np.int64)
    ds = FlatTensorDataset(x, y)
    assert len(ds) == 10