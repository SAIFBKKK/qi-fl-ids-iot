from __future__ import annotations

from pathlib import Path

from flwr.common.constant import PARTITION_ID_KEY

from src.common.paths import DATA_DIR
from src.fl.client.expert_client import ExpertClient
from src.fl.client.standard_client import StandardClient


def make_client_fn(config: dict):
    input_dim = int(config["dataset"]["feature_count"])
    output_dim = int(config["dataset"]["num_classes"])
    scenario_name = str(config["scenario"]["name"])
    batch_size = int(config["train"]["batch_size"])
    local_epochs = int(config["train"]["local_epochs"])
    learning_rate = float(config["train"]["learning_rate"])

    processed_root = DATA_DIR / "processed" / scenario_name

    def client_fn(context):
        partition_id = int(context.node_config.get(PARTITION_ID_KEY, context.node_id))
        node_idx = partition_id + 1
        node_name = f"node{node_idx}"

        train_path = processed_root / node_name / "train_preprocessed.npz"
        val_path = processed_root / node_name / "val_preprocessed.npz"

        is_expert = (
            bool(config.get("scenario", {}).get("expert_client_enabled", False))
            and node_name == str(config["scenario"].get("expert_client_id", "node3"))
        )

        cls = ExpertClient if is_expert else StandardClient

        return cls(
            client_id=node_name,
            train_path=train_path,
            val_path=val_path,
            input_dim=input_dim,
            output_dim=output_dim,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
        ).to_client()

    return client_fn
