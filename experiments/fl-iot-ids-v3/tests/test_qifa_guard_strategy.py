from __future__ import annotations

import numpy as np

from src.fl.qifa_guard_strategy import (
    QIFAGuardClientUpdate,
    QIFAGuardConfig,
    aggregate_qifa_guard_ndarrays,
    compute_qifa_guard_client_weights,
)


def _updates(num_clients: int = 3) -> list[QIFAGuardClientUpdate]:
    updates = []
    for idx in range(num_clients):
        scale = float(idx + 1)
        updates.append(
            QIFAGuardClientUpdate(
                parameters=[
                    np.full((2, 3), scale, dtype=np.float32),
                    np.full((3,), scale + 0.25, dtype=np.float32),
                ],
                num_examples=(idx + 1) * 10,
                node_id=f"node{idx + 1}",
                global_val_loss=0.5,
                global_rare_recall=0.2,
            )
        )
    return updates


def _fedavg(updates: list[QIFAGuardClientUpdate]) -> list[np.ndarray]:
    total = sum(update.num_examples for update in updates)
    return [
        sum(
            (update.num_examples / total) * update.parameters[layer_idx]
            for update in updates
        )
        for layer_idx in range(len(updates[0].parameters))
    ]


def test_qifa_guard_output_shape_matches_fedavg():
    updates = _updates(3)

    result, _ = aggregate_qifa_guard_ndarrays(
        updates,
        config=QIFAGuardConfig(lambda_qifa=0.2, beta_loss=1.0, rho_rare=0.5),
        server_round=1,
    )
    baseline = _fedavg(updates)

    assert [array.shape for array in result] == [array.shape for array in baseline]


def test_qifa_guard_lambda_zero_beta_zero_rho_zero_no_clip_matches_fedavg():
    updates = _updates(3)

    result, _ = aggregate_qifa_guard_ndarrays(
        updates,
        config=QIFAGuardConfig(
            lambda_qifa=0.0,
            beta_loss=0.0,
            rho_rare=0.0,
            min_client_weight=None,
            max_client_weight=None,
        ),
        server_round=1,
    )
    baseline = _fedavg(updates)

    for guard_layer, fedavg_layer in zip(result, baseline):
        assert np.allclose(guard_layer, fedavg_layer)


def test_qifa_guard_weights_normalized_to_one():
    updates = _updates(3)

    _, _, _, weights = compute_qifa_guard_client_weights(
        updates,
        config=QIFAGuardConfig(lambda_qifa=0.3, beta_loss=0.7, rho_rare=0.4),
    )

    assert np.isclose(weights.sum(), 1.0)


def test_qifa_guard_max_client_weight_respected():
    updates = [
        QIFAGuardClientUpdate(
            parameters=[np.asarray([1.0], dtype=np.float32)],
            num_examples=100,
            node_id="node1",
            global_val_loss=0.01,
            global_rare_recall=1.0,
        ),
        QIFAGuardClientUpdate(
            parameters=[np.asarray([2.0], dtype=np.float32)],
            num_examples=1,
            node_id="node2",
            global_val_loss=2.0,
            global_rare_recall=0.0,
        ),
    ]

    _, _, _, weights = compute_qifa_guard_client_weights(
        updates,
        config=QIFAGuardConfig(
            beta_loss=1.0,
            rho_rare=1.0,
            max_client_weight=0.6,
        ),
    )

    assert np.all(weights <= 0.6 + 1e-12)
    assert np.isclose(weights.sum(), 1.0)


def test_qifa_guard_rare_bonus_increases_weight():
    updates = [
        QIFAGuardClientUpdate(
            parameters=[np.asarray([1.0], dtype=np.float32)],
            num_examples=10,
            node_id="low_rare",
            global_val_loss=0.5,
            global_rare_recall=0.0,
        ),
        QIFAGuardClientUpdate(
            parameters=[np.asarray([1.0], dtype=np.float32)],
            num_examples=10,
            node_id="high_rare",
            global_val_loss=0.5,
            global_rare_recall=1.0,
        ),
    ]

    _, _, _, weights = compute_qifa_guard_client_weights(
        updates,
        config=QIFAGuardConfig(rho_rare=0.5),
    )

    assert weights[1] > weights[0]


def test_qifa_guard_high_loss_decreases_weight():
    updates = [
        QIFAGuardClientUpdate(
            parameters=[np.asarray([1.0], dtype=np.float32)],
            num_examples=10,
            node_id="low_loss",
            global_val_loss=0.1,
            global_rare_recall=0.0,
        ),
        QIFAGuardClientUpdate(
            parameters=[np.asarray([1.0], dtype=np.float32)],
            num_examples=10,
            node_id="high_loss",
            global_val_loss=2.0,
            global_rare_recall=0.0,
        ),
    ]

    _, _, _, weights = compute_qifa_guard_client_weights(
        updates,
        config=QIFAGuardConfig(beta_loss=1.0),
    )

    assert weights[0] > weights[1]


def test_qifa_guard_deterministic_seed():
    updates = _updates(3)
    config = QIFAGuardConfig(
        lambda_qifa=0.2,
        beta_loss=1.0,
        rho_rare=0.5,
        random_seed=77,
    )

    first, first_metrics = aggregate_qifa_guard_ndarrays(
        updates,
        config=config,
        server_round=4,
    )
    second, second_metrics = aggregate_qifa_guard_ndarrays(
        updates,
        config=config,
        server_round=4,
    )

    for first_layer, second_layer in zip(first, second):
        assert np.allclose(first_layer, second_layer)
    assert first_metrics == second_metrics
