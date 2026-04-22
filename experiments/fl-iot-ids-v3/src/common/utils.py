def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


EXPECTED_NODE_IDS = ["node1", "node2", "node3"]


def get_expected_node_ids(num_clients: int) -> list[str]:
    node_ids = list(EXPECTED_NODE_IDS)
    assert len(node_ids) == num_clients, (
        f"Expected exactly {num_clients} clients, got node_ids={node_ids}"
    )
    assert len(set(node_ids)) == len(node_ids), (
        f"Duplicate node_ids detected: {node_ids}"
    )
    return node_ids


def resolve_node_id_from_partition(partition_id: int, node_ids: list[str]) -> str:
    if partition_id < 0 or partition_id >= len(node_ids):
        raise ValueError(
            f"Invalid partition_id={partition_id} for node_ids={node_ids}"
        )
    return node_ids[partition_id]
