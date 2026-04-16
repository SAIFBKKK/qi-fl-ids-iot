from __future__ import annotations

import time
from typing import Iterable

import numpy as np


def now_perf() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return float(time.perf_counter() - start)


def parameters_size_bytes(parameters: Iterable[np.ndarray]) -> int:
    """
    Total serialized size approximation from ndarray buffers.
    """
    total = 0
    for arr in parameters:
        total += int(arr.nbytes)
    return total