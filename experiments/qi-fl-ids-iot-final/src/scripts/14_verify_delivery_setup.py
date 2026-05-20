from __future__ import annotations

import json
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = FINAL_DIR / "deployment"
L1_DIR = DEPLOYMENT_DIR / "l1_final"
API_DIR = DEPLOYMENT_DIR / "final_ids_api"
DASHBOARD_DIR = FINAL_DIR / "dashboard"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        L1_DIR / "deployment_manifest.json",
        L1_DIR / "selected_model.json",
        L1_DIR / "model_registry_deployment.json",
        L1_DIR / "qga_mask_reference.json",
        L1_DIR / "feature_schema.json",
        API_DIR / "app.py",
        API_DIR / "model_loader.py",
        API_DIR / "Dockerfile",
        DASHBOARD_DIR / "app.py",
        DASHBOARD_DIR / "Dockerfile",
        DEPLOYMENT_DIR / "docker-compose.final.yml",
        FINAL_DIR / "docs" / "14_docker_stack_final_delivery.md",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing:{path}")

    manifest = _read_json(L1_DIR / "deployment_manifest.json")
    selected = _read_json(L1_DIR / "selected_model.json")
    if selected.get("model_id") != "p8_fedavg_qga_l1":
        errors.append("selected_model_not_p8_fedavg_qga_l1")
    if selected.get("selected_mask_id") != "conservative_seed_42":
        errors.append("selected_mask_id_not_conservative_seed_42")
    if manifest.get("test_scaled_npz_included") is not False:
        errors.append("test_scaled_npz_must_not_be_included")
    if manifest.get("status") != "ready":
        warnings.append(f"model_bundle_status={manifest.get('status', 'unknown')}")

    forbidden_names = {"test_scaled.npz", "train_scaled.npz", "val_scaled.npz", "train.parquet", "test.parquet", "val.parquet"}
    included_forbidden = [
        str(path)
        for path in DEPLOYMENT_DIR.rglob("*")
        if path.is_file() and path.name in forbidden_names
    ]
    if included_forbidden:
        errors.append(f"forbidden_dataset_files_in_deployment:{included_forbidden}")

    accepted = not errors
    payload = {"accepted": accepted, "errors": errors, "warnings": warnings}
    print(json.dumps(payload, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
