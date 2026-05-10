from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.label_mapping import (  # noqa: E402
    BENIGN_LABEL_ID,
    BENIGN_LABEL_NAME,
    EXPECTED_FEATURES,
    LABEL_COLUMN,
    build_label_to_binary,
    build_label_to_family,
    load_label_mapping,
)

FINAL_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
DATA_DIR = REPO_ROOT / "data" / "balancing_v3_fixed300k_outputs"
PARQUET_PATH = DATA_DIR / "balancing_v3_fixed300k_balanced.parquet"
LABEL_MAPPING_PATH = DATA_DIR / "label_mapping.json"


@pytest.fixture(scope="module")
def label_mapping() -> dict[str, int]:
    return load_label_mapping(LABEL_MAPPING_PATH)


def test_label_mapping_has_34_classes(label_mapping: dict[str, int]) -> None:
    assert len(label_mapping) == 34


def test_benign_label_id_is_1(label_mapping: dict[str, int]) -> None:
    assert label_mapping[BENIGN_LABEL_NAME] == BENIGN_LABEL_ID


def test_family_mapping_covers_all_labels(label_mapping: dict[str, int]) -> None:
    label_to_family = build_label_to_family(label_mapping)
    assert set(label_to_family) == set(label_mapping)


def test_binary_mapping_covers_all_labels(label_mapping: dict[str, int]) -> None:
    label_to_binary = build_label_to_binary(label_mapping)
    assert set(label_to_binary) == set(label_mapping)
    assert label_to_binary[BENIGN_LABEL_NAME]["binary_label"] == 0
    assert all(
        payload["binary_label"] == 1
        for label, payload in label_to_binary.items()
        if label != BENIGN_LABEL_NAME
    )


def test_expected_features_count_is_28() -> None:
    assert len(EXPECTED_FEATURES) == 28


def test_label_column_is_excluded_from_features() -> None:
    assert LABEL_COLUMN not in EXPECTED_FEATURES


def test_parquet_metadata_loads() -> None:
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    assert parquet_file.metadata.num_rows == 9_401_350
    assert parquet_file.metadata.num_columns == 29


def test_validation_outputs_created() -> None:
    expected_outputs = [
        FINAL_DIR / "outputs" / "reports" / "data_validation_summary.json",
        FINAL_DIR / "outputs" / "reports" / "data_validation_profile.json",
        FINAL_DIR / "outputs" / "artifacts" / "features" / "feature_names.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_mapping.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "id_to_label.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_to_family.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_to_binary.json",
        FINAL_DIR / "docs" / "01_data_validation.md",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "01_binary_distribution.png",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "02_class_distribution_34.png",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "03_family_distribution.png",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "04_missing_values_by_column.png",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "05_infinite_values_by_column.png",
        FINAL_DIR / "outputs" / "figures" / "data_validation" / "06_feature_types.png",
    ]
    missing = [path for path in expected_outputs if not path.exists()]
    if missing:
        pytest.skip("P1 validation outputs have not been generated yet")
    assert all(path.stat().st_size > 0 for path in expected_outputs)
