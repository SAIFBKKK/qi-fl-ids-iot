from __future__ import annotations

import json
import sys
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[2]
API_DIR = FINAL_DIR / "deployment" / "final_ids_api"


def main() -> int:
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    health = client.get("/health")
    ready = client.get("/ready")
    info = client.get("/model/info")
    errors: list[str] = []
    warnings: list[str] = []
    if health.status_code != 200 or health.json().get("status") != "ok":
        errors.append("health_failed")
    if ready.status_code != 200:
        errors.append("ready_endpoint_failed")
    if info.status_code != 200:
        errors.append("model_info_failed")

    selected_count = int(info.json().get("selected_model", {}).get("features_count", 12)) if info.status_code == 200 else 12
    if ready.status_code == 200 and ready.json().get("ready") is True:
        response = client.post("/predict", json={"features": [0.0] * selected_count})
        if response.status_code != 200:
            errors.append(f"predict_failed:{response.status_code}:{response.text}")
        else:
            payload = response.json()
            if payload.get("model_id") != "p8_fedavg_qga_l1":
                errors.append("predict_wrong_model_id")
    else:
        warnings.append(ready.json().get("message", "model artifact unavailable") if ready.status_code == 200 else "ready_failed")

    accepted = not errors
    print(json.dumps({"accepted": accepted, "errors": errors, "warnings": warnings}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
