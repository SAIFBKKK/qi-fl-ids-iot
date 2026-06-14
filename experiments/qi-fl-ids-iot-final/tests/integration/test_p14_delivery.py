from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = FINAL_DIR / "deployment"
L1_DIR = DEPLOYMENT_DIR / "l1_final"
API_DIR = DEPLOYMENT_DIR / "final_ids_api"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p14_bundle_manifest_and_selected_model() -> None:
    manifest = _read_json(L1_DIR / "deployment_manifest.json")
    selected = _read_json(L1_DIR / "selected_model.json")
    assert manifest["selected_model"] == "P8 FedAvg + QGA"
    assert manifest["selected_mask_id"] == "conservative_seed_42"
    assert selected["model_id"] == "p8_fedavg_qga_l1"
    assert selected["selected_mask_id"] == "conservative_seed_42"
    assert selected["features_count"] == 12
    assert manifest["test_scaled_npz_included"] is False


def test_p14_required_files_exist() -> None:
    required = [
        API_DIR / "app.py",
        API_DIR / "model_loader.py",
        API_DIR / "preprocessor.py",
        API_DIR / "schemas.py",
        API_DIR / "metrics.py",
        API_DIR / "Dockerfile",
        FINAL_DIR / "dashboard" / "Dockerfile",
        DEPLOYMENT_DIR / "docker-compose.final.yml",
        FINAL_DIR / "docs" / "14_docker_stack_final_delivery.md",
        FINAL_DIR / "outputs" / "reports" / "p14_docker_delivery_audit.md",
        FINAL_DIR / "outputs" / "reports" / "p14_docker_delivery_plan.md",
    ]
    for path in required:
        assert path.exists(), path


def test_p14_no_heavy_datasets_in_deployment() -> None:
    forbidden = {"train_scaled.npz", "val_scaled.npz", "test_scaled.npz", "train.parquet", "val.parquet", "test.parquet"}
    found = [path for path in DEPLOYMENT_DIR.rglob("*") if path.is_file() and path.name in forbidden]
    assert found == []


def test_p14_verify_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(FINAL_DIR / "src" / "scripts" / "14_verify_delivery_setup.py")],
        cwd=FINAL_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["accepted"] is True


def test_p14_model_artifact_warning_or_packaged() -> None:
    manifest = _read_json(L1_DIR / "deployment_manifest.json")
    status = manifest["artifacts"]["model"]["status"]
    assert status in {"packaged", "model_checkpoint_missing", "model_checkpoint_too_large"}
    if status == "packaged":
        assert (L1_DIR / "artifacts" / "model.pth").exists()
        assert manifest["artifacts"]["model"]["sha256"]
