from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
PARTITIONS_DIR = FINAL_DIR / "outputs" / "partitions"
ALPHAS = ["alpha_0.1", "alpha_0.5", "alpha_5.0"]
CLIENTS = [3, 4, 5]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scenario_dirs(dataset_level: str) -> list[Path]:
    return [
        PARTITIONS_DIR / dataset_level / alpha / f"k{k}"
        for alpha in ALPHAS
        for k in CLIENTS
    ]


def _manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("P3 Dirichlet outputs have not been generated yet")
    return _load_json(manifest_path)


def test_p2_outputs_exist() -> None:
    expected = [
        FINAL_DIR / "docs" / "02_preprocessing.md",
        FINAL_DIR / "outputs" / "preprocessed" / "l1_binary" / "train_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l1_binary" / "val_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l1_binary" / "test_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l1_binary" / "manifest.json",
        FINAL_DIR / "outputs" / "preprocessed" / "l2_family" / "train_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l2_family" / "val_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l2_family" / "test_scaled.npz",
        FINAL_DIR / "outputs" / "preprocessed" / "l2_family" / "manifest.json",
    ]
    assert all(path.exists() for path in expected)


def test_l1_all_scenarios_exist() -> None:
    assert all(path.exists() for path in _scenario_dirs("l1_binary"))


def test_l2_all_scenarios_exist() -> None:
    assert all(path.exists() for path in _scenario_dirs("l2_family"))


def test_l1_global_test_not_partitioned() -> None:
    expected_test = "experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/test_scaled.npz"
    for scenario_dir in _scenario_dirs("l1_binary"):
        manifest = _manifest(scenario_dir)
        assert manifest["partition_test"] is False
        assert manifest["source_global_test_npz"] == expected_test
        assert (scenario_dir / "global_test_reference.json").exists()
        assert not list(scenario_dir.glob("client_*/test_scaled.npz"))


def test_l2_global_test_not_partitioned() -> None:
    expected_test = "experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/test_scaled.npz"
    for scenario_dir in _scenario_dirs("l2_family"):
        manifest = _manifest(scenario_dir)
        assert manifest["partition_test"] is False
        assert manifest["source_global_test_npz"] == expected_test
        assert (scenario_dir / "global_test_reference.json").exists()
        assert not list(scenario_dir.glob("client_*/test_scaled.npz"))


def test_l1_client_npz_train_val_exist() -> None:
    for scenario_dir in _scenario_dirs("l1_binary"):
        manifest = _manifest(scenario_dir)
        for files in manifest["client_files"].values():
            assert (REPO_ROOT / files["train_scaled_npz"]).exists()
            assert (REPO_ROOT / files["val_scaled_npz"]).exists()


def test_l2_index_only_files_exist() -> None:
    for scenario_dir in _scenario_dirs("l2_family"):
        manifest = _manifest(scenario_dir)
        assert manifest["storage_mode"] == "index_only"
        for files in manifest["client_files"].values():
            assert (REPO_ROOT / files["train_row_ids_npy"]).exists()
            assert (REPO_ROOT / files["val_row_ids_npy"]).exists()
        assert not list(scenario_dir.glob("client_*/*scaled.npz"))


def test_no_client_empty() -> None:
    for dataset_level in ["l1_binary", "l2_family"]:
        for scenario_dir in _scenario_dirs(dataset_level):
            manifest = _manifest(scenario_dir)
            for counts in manifest["row_counts"]["by_client"].values():
                assert counts["train"] > 0
                assert counts["val"] > 0


def test_l1_train_union_matches_global_count() -> None:
    for scenario_dir in _scenario_dirs("l1_binary"):
        manifest = _manifest(scenario_dir)
        assert manifest["row_counts"]["train_total"] == 441_000
        assert manifest["anti_leakage_result"]["train_union_matches_global_count"] is True


def test_l1_val_union_matches_global_count() -> None:
    for scenario_dir in _scenario_dirs("l1_binary"):
        manifest = _manifest(scenario_dir)
        assert manifest["row_counts"]["val_total"] == 94_500
        assert manifest["anti_leakage_result"]["val_union_matches_global_count"] is True


def test_l2_train_union_matches_global_count() -> None:
    for scenario_dir in _scenario_dirs("l2_family"):
        manifest = _manifest(scenario_dir)
        assert manifest["row_counts"]["train_total"] == 6_370_944
        assert manifest["anti_leakage_result"]["train_union_matches_global_count"] is True


def test_l2_val_union_matches_global_count() -> None:
    for scenario_dir in _scenario_dirs("l2_family"):
        manifest = _manifest(scenario_dir)
        assert manifest["row_counts"]["val_total"] == 1_365_202
        assert manifest["anti_leakage_result"]["val_union_matches_global_count"] is True


def test_no_overlap_between_clients() -> None:
    for dataset_level in ["l1_binary", "l2_family"]:
        for scenario_dir in _scenario_dirs(dataset_level):
            manifest = _manifest(scenario_dir)
            anti_leakage = manifest["anti_leakage_result"]
            assert anti_leakage["no_overlap_between_clients"] is True
            assert anti_leakage["no_train_val_leakage"] is True
            assert anti_leakage["valid"] is True


def test_manifest_has_global_test_reference() -> None:
    for dataset_level in ["l1_binary", "l2_family"]:
        for scenario_dir in _scenario_dirs(dataset_level):
            manifest = _manifest(scenario_dir)
            reference = REPO_ROOT / manifest["global_test_reference_file"]
            assert reference.exists()
            payload = _load_json(reference)
            assert payload["keep_global_test_holdout"] is True
            assert payload["partition_test"] is False


def test_distribution_reports_exist() -> None:
    for dataset_level in ["l1_binary", "l2_family"]:
        for scenario_dir in _scenario_dirs(dataset_level):
            assert (scenario_dir / "distribution_report.json").exists()
            assert (scenario_dir / "client_distribution.csv").exists()
