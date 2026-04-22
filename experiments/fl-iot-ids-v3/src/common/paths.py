from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MLRUNS_DIR = OUTPUTS_DIR / "mlruns"

# Dataset parameters — v3
INPUT_DIM = 28
NUM_CLASSES = 34
BENIGN_CLASS = 1  # BenignTraffic=label_id 1 in v3 dataset (was 0 in v1)


def get_raw_path(scenario: str, node_id: str) -> Path:
    """Return the raw CSV path for a given scenario and node."""
    return RAW_DIR / scenario / node_id / "train.csv"


def get_processed_path(scenario: str, node_id: str) -> Path:
    """Return the preprocessed NPZ path for a given scenario and node."""
    return PROCESSED_DIR / scenario / node_id / "train_preprocessed.npz"

DATASET_CSV = Path(
    "E:/dataset/CICIoT2023/balancing_v3_fixed300k_outputs"
    "/balancing_v3_fixed300k_balanced.csv"
)
DATASET_PARQUET = Path(
    "E:/dataset/CICIoT2023/balancing_v3_fixed300k_outputs"
    "/balancing_v3_fixed300k_balanced.parquet"
)


def ensure_runtime_dirs() -> None:
    for path in [
        OUTPUTS_DIR / "logs",
        OUTPUTS_DIR / "metrics",
        OUTPUTS_DIR / "reports",
        OUTPUTS_DIR / "checkpoints",
        MLRUNS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
