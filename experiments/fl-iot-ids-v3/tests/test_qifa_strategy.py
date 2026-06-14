from __future__ import annotations

import numpy as np

from src.fl.qifa_strategy import QIFAClientUpdate, QIFAConfig, aggregate_qifa_ndarrays


def _updates(num_clients: int) -> list[QIFAClientUpdate]:
    updates = []
    for idx in range(num_clients):
        scale = float(idx + 1)
        updates.append(
            QIFAClientUpdate(
                parameters=[
                    np.full((2, 3), scale, dtype=np.float32),
                    np.full((3,), scale + 0.5, dtype=np.float32),
                ],
                num_examples=(idx + 1) * 10,
                node_id=f"node{idx + 1}",
                epsilon=1.0 + 0.1 * idx,
            )
        )
    return updates


def _fedavg(updates: list[QIFAClientUpdate]) -> list[np.ndarray]:
    total = sum(update.num_examples for update in updates)
    return [
        sum(
            (update.num_examples / total) * update.parameters[layer_idx]
            for update in updates
        )
        for layer_idx in range(len(updates[0].parameters))
    ]


def test_qifa_output_shape_matches_fedavg_output_shape():
    updates = _updates(3)
    result, _ = aggregate_qifa_ndarrays(
        updates,
        config=QIFAConfig(lambda_qifa=0.3),
        server_round=1,
    )
    baseline = _fedavg(updates)

    assert [array.shape for array in result] == [array.shape for array in baseline]


def test_qifa_works_with_one_two_and_three_clients():
    for num_clients in (1, 2, 3):
        result, metrics = aggregate_qifa_ndarrays(
            _updates(num_clients),
            config=QIFAConfig(lambda_qifa=0.2),
            server_round=1,
        )

        assert len(result) == 2
        assert metrics["qifa_effective_clients"] == float(num_clients)


def test_qifa_lambda_zero_without_perturbation_matches_fedavg():
    updates = _updates(3)
    result, metrics = aggregate_qifa_ndarrays(
        updates,
        config=QIFAConfig(lambda_qifa=0.0, perturbation_enabled=False),
        server_round=1,
    )
    baseline = _fedavg(updates)

    for qifa_layer, fedavg_layer in zip(result, baseline):
        assert np.allclose(qifa_layer, fedavg_layer)
    assert metrics["qifa_perturbation_norm"] == 0.0
    assert metrics["qifa/perturbation_applied"] == 0.0


def test_qifa_perturbation_can_be_disabled():
    updates = _updates(2)
    common = dict(lambda_qifa=0.1, delta_perturbation=1.0, sigma_noise=0.2)
    disabled, disabled_metrics = aggregate_qifa_ndarrays(
        updates,
        config=QIFAConfig(**common, perturbation_enabled=False, random_seed=123),
        server_round=1,
    )
    enabled, enabled_metrics = aggregate_qifa_ndarrays(
        updates,
        config=QIFAConfig(**common, perturbation_enabled=True, random_seed=123),
        server_round=1,
    )

    assert disabled_metrics["qifa_perturbation_norm"] == 0.0
    assert disabled_metrics["qifa/perturbation_applied"] == 0.0
    assert enabled_metrics["qifa_perturbation_norm"] > 0.0
    assert enabled_metrics["qifa/perturbation_applied"] == 1.0
    assert any(not np.allclose(a, b) for a, b in zip(disabled, enabled))


def test_qifa_deterministic_seed_reproduces_perturbation():
    updates = _updates(3)
    cfg = QIFAConfig(
        lambda_qifa=0.2,
        perturbation_enabled=True,
        delta_perturbation=0.5,
        sigma_noise=0.1,
        random_seed=77,
    )

    first, first_metrics = aggregate_qifa_ndarrays(updates, config=cfg, server_round=4)
    second, second_metrics = aggregate_qifa_ndarrays(updates, config=cfg, server_round=4)

    for first_layer, second_layer in zip(first, second):
        assert np.allclose(first_layer, second_layer)
    assert first_metrics == second_metrics
