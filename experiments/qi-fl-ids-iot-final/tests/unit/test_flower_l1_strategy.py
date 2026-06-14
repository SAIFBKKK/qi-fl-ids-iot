from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from flwr.server.strategy import FedAvg

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_l1_flower.communication import round_bandwidth  # noqa: E402
from fl_l1_flower.metrics import aggregate_evaluate_metrics  # noqa: E402
from fl_l1_flower.strategy import FlowerL1FedAvgStrategy  # noqa: E402
from fl_l1_flower.task import build_model, get_parameters, set_parameters  # noqa: E402


def test_strategy_is_fedavg() -> None:
    assert issubclass(FlowerL1FedAvgStrategy, FedAvg)


def test_model_parameters_conversion() -> None:
    cfg = {
        "input_dim": 28,
        "hidden_layers": [128, 64],
        "output_dim": 2,
        "dropout": 0.2,
        "activation": "relu",
    }
    model = build_model(cfg)
    params = get_parameters(model)
    assert params
    shifted = [np.asarray(array) + np.float32(0.0) for array in params]
    set_parameters(model, shifted)
    assert [tuple(array.shape) for array in get_parameters(model)] == [tuple(array.shape) for array in shifted]


def test_weighted_metrics_aggregation() -> None:
    metrics = aggregate_evaluate_metrics(
        [
            (10, {"TP": 8, "TN": 1, "FP": 1, "FN": 0, "local_val_loss": 0.2}),
            (20, {"TP": 10, "TN": 5, "FP": 2, "FN": 3, "local_val_loss": 0.4}),
        ]
    )
    assert metrics["TP"] == 18
    assert metrics["TN"] == 6
    assert metrics["FP"] == 3
    assert metrics["FN"] == 3
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_bandwidth_formula() -> None:
    result = round_bandwidth(model_size_bytes_value=48_392, num_clients=3, previous_cumulative_bytes=0)
    assert result["upload_bytes"] == 145_176
    assert result["download_bytes"] == 145_176
    assert result["total_bytes"] == 290_352


def test_bandwidth_cumulative_formula() -> None:
    first = round_bandwidth(model_size_bytes_value=48_392, num_clients=3, previous_cumulative_bytes=0)
    second = round_bandwidth(
        model_size_bytes_value=48_392,
        num_clients=3,
        previous_cumulative_bytes=int(first["cumulative_bytes"]),
    )
    assert second["cumulative_bytes"] == first["total_bytes"] + second["total_bytes"]
