from __future__ import annotations

import numpy as np
import pickle

import src.fl.client_app as client_app
from src.fl.client_app import FlowerClient, get_model_parameters
from src.model.validation import resolve_num_classes


def _write_client_npz(root, scenario: str, node_id: str, labels: np.ndarray) -> None:
    node_dir = root / scenario / node_id
    node_dir.mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(42).normal(size=(len(labels), 28)).astype(np.float32)
    for split in ("train", "val"):
        np.savez_compressed(
            node_dir / f"{split}_preprocessed.npz",
            X=X,
            y=labels.astype(np.int64),
            feature_names=np.array([f"f{i}" for i in range(28)], dtype=object),
        )


def test_two_clients_keep_identical_parameter_shapes_with_missing_classes(tmp_path, monkeypatch):
    scenario = "synthetic_noniid"
    _write_client_npz(tmp_path, scenario, "node1", np.array([0, 1, 1, 0, 1, 0]))
    _write_client_npz(tmp_path, scenario, "node2", np.array([2, 3, 3, 2, 3, 2]))

    monkeypatch.setattr(
        client_app,
        "get_processed_path",
        lambda scenario_name, node_id: tmp_path / scenario_name / node_id / "train_preprocessed.npz",
    )

    common_kwargs = {
        "scenario": scenario,
        "batch_size": 2,
        "local_epochs": 1,
        "learning_rate": 1e-3,
        "num_classes": 34,
        "hidden_dims": (8, 4),
        "imbalance_strategy": "none",
    }
    c1 = FlowerClient(node_id="node1", **common_kwargs)
    c2 = FlowerClient(node_id="node2", **common_kwargs)

    shapes_1 = [param.shape for param in get_model_parameters(c1.model)]
    shapes_2 = [param.shape for param in get_model_parameters(c2.model)]

    assert shapes_1 == shapes_2
    assert shapes_1[-1] == (34,)


def test_num_classes_is_resolved_from_dataset_config_not_local_labels():
    assert resolve_num_classes({"num_classes": 34}, {"output_dim": 34}) == 34


def _flatten(parameters):
    return np.concatenate([param.reshape(-1) for param in parameters])


def test_fedavg_fedprox_and_scaffold_produce_distinct_synthetic_updates(tmp_path, monkeypatch):
    scenario = "synthetic_strategy"
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    _write_client_npz(tmp_path, scenario, "node1", labels)

    monkeypatch.setattr(
        client_app,
        "get_processed_path",
        lambda scenario_name, node_id: tmp_path / scenario_name / node_id / "train_preprocessed.npz",
    )

    common_kwargs = {
        "node_id": "node1",
        "scenario": scenario,
        "batch_size": 2,
        "local_epochs": 2,
        "learning_rate": 1e-2,
        "num_classes": 34,
        "hidden_dims": (8, 4),
        "dropout": 0.0,
        "imbalance_strategy": "none",
    }
    template = FlowerClient(**common_kwargs, fl_strategy="fedavg")
    initial = [param.copy() for param in get_model_parameters(template.model)]

    fedavg = FlowerClient(**common_kwargs, fl_strategy="fedavg")
    fedprox = FlowerClient(**common_kwargs, fl_strategy="fedprox", proximal_mu=1.0)
    scaffold = FlowerClient(**common_kwargs, fl_strategy="scaffold")

    avg_params, _, _ = fedavg.fit([param.copy() for param in initial], {})
    prox_params, _, _ = fedprox.fit([param.copy() for param in initial], {})
    c_global = [np.full_like(param, 0.01, dtype=np.float32) for param in initial]
    scaffold_params, _, scaffold_metrics = scaffold.fit(
        [param.copy() for param in initial],
        {"scaffold_c_global": pickle.dumps(c_global)},
    )

    assert "scaffold_delta_c" in scaffold_metrics
    assert not np.allclose(_flatten(avg_params), _flatten(prox_params))
    assert not np.allclose(_flatten(avg_params), _flatten(scaffold_params))


def test_scaffold_persists_c_local_across_client_instances(tmp_path, monkeypatch):
    scenario = "synthetic_scaffold_persistence"
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    _write_client_npz(tmp_path, scenario, "node1", labels)

    monkeypatch.setattr(
        client_app,
        "get_processed_path",
        lambda scenario_name, node_id: tmp_path / scenario_name / node_id / "train_preprocessed.npz",
    )

    scaffold_state_dir = tmp_path / "scaffold_state" / "run-001"
    common_kwargs = {
        "node_id": "node1",
        "scenario": scenario,
        "batch_size": 2,
        "local_epochs": 1,
        "learning_rate": 1e-2,
        "num_classes": 34,
        "hidden_dims": (8, 4),
        "dropout": 0.0,
        "imbalance_strategy": "none",
        "fl_strategy": "scaffold",
        "scaffold_state_dir": scaffold_state_dir,
    }

    template = FlowerClient(**common_kwargs)
    initial = [param.copy() for param in get_model_parameters(template.model)]
    c_global = [np.full_like(param, 0.01, dtype=np.float32) for param in initial]

    first_client = FlowerClient(**common_kwargs)
    first_client.fit(
        [param.copy() for param in initial],
        {"scaffold_c_global": pickle.dumps(c_global)},
    )

    state_path = scaffold_state_dir / "c_local_node1.pkl"
    assert state_path.exists()

    with state_path.open("rb") as handle:
        first_saved = pickle.load(handle)

    assert any(not np.allclose(param, 0.0) for param in first_saved)

    second_client = FlowerClient(**common_kwargs)
    loaded_c_local = second_client._get_c_local()
    for saved, loaded in zip(first_saved, loaded_c_local):
        assert np.allclose(saved, loaded)

    second_client.fit(
        [param.copy() for param in initial],
        {"scaffold_c_global": pickle.dumps(c_global)},
    )

    with state_path.open("rb") as handle:
        second_saved = pickle.load(handle)

    assert any(
        not np.allclose(before, after)
        for before, after in zip(first_saved, second_saved)
    )
