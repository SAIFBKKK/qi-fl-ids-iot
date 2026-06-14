from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
L1_DIR = FINAL_DIR / "outputs" / "preprocessed" / "l1_binary"
L2_DIR = FINAL_DIR / "outputs" / "preprocessed" / "l2_family"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def l1_manifest() -> dict:
    path = L1_DIR / "manifest.json"
    if not path.exists():
        pytest.skip("P2 L1 manifest has not been generated yet")
    return _load_json(path)


@pytest.fixture(scope="module")
def l2_manifest() -> dict:
    path = L2_DIR / "manifest.json"
    if not path.exists():
        pytest.skip("P2 L2 manifest has not been generated yet")
    return _load_json(path)


@pytest.fixture(scope="module")
def l1_sampling_report() -> dict:
    path = L1_DIR / "sampling_report.json"
    if not path.exists():
        pytest.skip("P2 L1 sampling report has not been generated yet")
    return _load_json(path)


@pytest.fixture(scope="module")
def l2_distribution_report() -> dict:
    path = L2_DIR / "distribution_report.json"
    if not path.exists():
        pytest.skip("P2 L2 distribution report has not been generated yet")
    return _load_json(path)


def test_p1_artifacts_exist() -> None:
    expected = [
        FINAL_DIR / "docs" / "01_data_validation.md",
        FINAL_DIR / "outputs" / "artifacts" / "features" / "feature_names.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_mapping.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "id_to_label.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_to_binary.json",
        FINAL_DIR / "outputs" / "artifacts" / "mappings" / "label_to_family.json",
    ]
    assert all(path.exists() for path in expected)


def test_l1_expected_counts(l1_manifest: dict) -> None:
    counts = l1_manifest["row_counts"]
    assert counts["total"] == 630_000
    assert counts["normal"] == 300_000
    assert counts["attack"] == 330_000


def test_l1_binary_labels_are_0_1(l1_manifest: dict) -> None:
    assert set(l1_manifest["row_counts"]["by_binary"].keys()) == {"normal", "attack"}
    assert l1_manifest["row_counts"]["by_binary"]["normal"]["binary_label"] == 0
    assert l1_manifest["row_counts"]["by_binary"]["attack"]["binary_label"] == 1


def test_l1_attack_sampling_per_class_is_10000(l1_sampling_report: dict) -> None:
    after = l1_sampling_report["count_after_sampling_by_label_id"]
    attack_counts = [
        payload["count"]
        for label_id, payload in after.items()
        if int(label_id) != 1
    ]
    assert len(attack_counts) == 33
    assert all(count == 10_000 for count in attack_counts)


def test_l1_split_ratios(l1_manifest: dict) -> None:
    split_counts = l1_manifest["actual_split_counts"]
    assert split_counts == {"train": 441_000, "val": 94_500, "test": 94_500}


def test_l1_no_row_id_overlap(l1_manifest: dict) -> None:
    anti_leakage = l1_manifest["anti_leakage_result"]
    assert anti_leakage["anti_leakage_valid"] is True
    assert anti_leakage["overlap_counts"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }


def test_l2_attack_only(l2_distribution_report: dict) -> None:
    assert l2_distribution_report["attack_only"] is True
    assert "1" not in l2_distribution_report["distribution_by_original_label_id"]


def test_l2_no_sampling_total_count(l2_distribution_report: dict) -> None:
    assert l2_distribution_report["sampling"] is False
    assert l2_distribution_report["total_count"] == 9_101_350


def test_l2_expected_families(l2_distribution_report: dict) -> None:
    expected = {
        "DDoS",
        "DoS",
        "Recon",
        "Web-based",
        "BruteForce",
        "Spoofing",
        "Mirai",
        "Malware",
    }
    assert set(l2_distribution_report["distribution_by_family_name"]) == expected


def test_l2_split_ratios(l2_manifest: dict) -> None:
    counts = l2_manifest["actual_split_counts"]
    total = l2_manifest["row_counts"]["total"]
    assert total == 9_101_350
    assert abs(counts["train"] / total - 0.70) < 0.001
    assert abs(counts["val"] / total - 0.15) < 0.001
    assert abs(counts["test"] / total - 0.15) < 0.001


def test_l2_no_row_id_overlap(l2_manifest: dict) -> None:
    anti_leakage = l2_manifest["anti_leakage_result"]
    assert anti_leakage["anti_leakage_valid"] is True
    assert anti_leakage["overlap_counts"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }


def test_scaler_train_only_manifest(l1_manifest: dict, l2_manifest: dict) -> None:
    for manifest in [l1_manifest, l2_manifest]:
        assert manifest["scaling_train_only"] is True
        if manifest["scaling_applied"]:
            scaler_path = REPO_ROOT / manifest["scaler_path"]
            assert scaler_path.exists()
            assert scaler_path.stat().st_size > 0


def test_npz_shapes_match_feature_count(l1_manifest: dict, l2_manifest: dict) -> None:
    for manifest in [l1_manifest, l2_manifest]:
        feature_count = manifest["feature_count"]
        for split_name, shapes in manifest["npz_shapes"].items():
            split_rows = manifest["actual_split_counts"][split_name]
            assert shapes["X"] == [split_rows, feature_count]
            assert shapes["label_id_original"] == [split_rows]
            assert shapes["row_id"] == [split_rows]


def test_parquet_split_metadata_counts(l1_manifest: dict, l2_manifest: dict) -> None:
    for manifest in [l1_manifest, l2_manifest]:
        for split_name, info in manifest["output_files"]["parquet"].items():
            parquet_path = REPO_ROOT / info["path"]
            metadata = pq.ParquetFile(parquet_path).metadata
            assert metadata.num_rows == manifest["actual_split_counts"][split_name]
