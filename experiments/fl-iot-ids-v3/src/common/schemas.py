from dataclasses import dataclass

@dataclass
class NodeConfig:
    node_id: str
    raw_dir: str
    processed_dir: str
