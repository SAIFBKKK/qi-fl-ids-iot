from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
L1_DIR = FINAL_DIR / "outputs" / "preprocessed" / "l1_binary"
RUN_DIR = FINAL_DIR / "outputs" / "centralized_l1"
ARTIFACTS_DIR = RUN_DIR / "artifacts"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_outputs() -> None:
    if not (CHECKPOINTS_DIR / "best_model.pth").exists():
        pytest.skip("P4 centralized L1 outputs have not been generated yet")


def test_l1_npz_exists() -> None:
    expected = [
        L1_DIR / "train_scaled.npz",
        L1_DIR / "val_scaled.npz",
        L1_DIR / "test_scaled.npz",
    ]
    assert all(path.exists() for path in expected)


def test_best_model_checkpoint_exists() -> None:
    _require_outputs()
    assert (CHECKPOINTS_DIR / "best_model.pth").stat().st_size > 0


def test_last_model_checkpoint_exists() -> None:
    _require_outputs()
    assert (CHECKPOINTS_DIR / "last_model.pth").stat().st_size > 0


def test_threshold_json_exists() -> None:
    _require_outputs()
    payload = _load_json(ARTIFACTS_DIR / "threshold.json")
    assert payload["selection_split"] == "validation"
    assert payload["test_used_for_threshold"] is False


def test_metrics_test_json_exists() -> None:
    _require_outputs()
    payload = _load_json(ARTIFACTS_DIR / "metrics_test.json")
    assert "macro_f1" in payload
    assert "TP" in payload
    assert "TN" in payload
    assert "FP" in payload
    assert "FN" in payload


def test_confusion_matrix_shape() -> None:
    _require_outputs()
    with (ARTIFACTS_DIR / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert set(rows[0]) == {"label", "pred_normal", "pred_attack"}


def test_training_history_created() -> None:
    _require_outputs()
    path = ARTIFACTS_DIR / "training_history.csv"
    assert path.exists()
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) >= 1
    assert "val_macro_f1" in rows[0]


def test_historical_kaggle_artifact_exists() -> None:
    _require_outputs()
    payload = _load_json(ARTIFACTS_DIR / "historical_kaggle_34class_baseline.json")
    assert payload["not_used_for_p4_l1"] is True
    assert payload["architecture"] == "28 -> 128 -> 64 -> 34"


def test_no_test_used_for_threshold_in_manifest_or_report() -> None:
    _require_outputs()
    threshold = _load_json(ARTIFACTS_DIR / "threshold.json")
    model_config = _load_json(ARTIFACTS_DIR / "model_config.json")
    report = (FINAL_DIR / "docs" / "04_centralized_baseline.md").read_text(encoding="utf-8")
    assert threshold["selection_split"] == "validation"
    assert threshold["test_used_for_threshold"] is False
    assert model_config["threshold_selection"]["split"] == "validation"
    assert model_config["threshold_selection"]["test_used_for_threshold"] is False
    assert "test global n’est pas utilisé" in report or "test global n'est pas utilisé" in report
