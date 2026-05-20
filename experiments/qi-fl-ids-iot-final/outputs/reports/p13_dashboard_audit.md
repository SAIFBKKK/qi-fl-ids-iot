# P13 Dashboard L1 Final Audit

## Scope

Branch: `final/quantum-inspired-fl-iot-ids-final`

Audited areas:
- `services/`
- `services/iot-node/`
- `services/traffic-generator/`
- `services/docker-compose.yml`
- `services/monitoring/`
- `experiments/qi-fl-ids-iot-final/`
- `experiments/qi-fl-ids-iot-final/outputs/reports/`
- `experiments/qi-fl-ids-iot-final/outputs/figures/`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/`

## Existing Dashboard

An existing FastAPI dashboard is present under `services/dashboard/`. It is designed for the service stack and exposes UI tabs such as IoT, FL, QI, and monitoring. Its `main.py` mounts `/static`, serves `/tab/{tab_name}`, and includes API routers for nodes, models, metrics, QI, scenarios, and system status.

This dashboard is coupled to service URLs such as `FL_SERVER_URL`, Prometheus, and MLflow. It is useful for P14 deployment work, but it should not be modified for P13 because the current objective is a final L1 evidence and evaluation dashboard without touching Docker or the existing service stack.

## Existing IDS Services

The repository contains service components for:
- dashboard
- IoT node
- traffic generation
- monitoring
- Docker composition

These services were audited in read-only mode only. No Docker, dashboard service, microservice, or monitoring file was modified.

## Existing Endpoints

The service dashboard exposes:
- `GET /`
- `GET /tab/{tab_name}`
- `POST /api/connect`
- `GET /api/fl/runs`
- `GET /api/fl/schedule`
- `POST /api/fl/trigger`
- `GET /api/fl/health`
- `GET /health`

Additional routers are mounted from `api.models`, `api.nodes`, `api.metrics`, `api.qi`, `api.scenarios`, and `api.system`.

## Model Artifacts Available Locally

Local model artifacts/checkpoints exist under the final experiment outputs, including:
- centralized L1 checkpoints
- P5 FedAvg L1 checkpoints
- P5.2 Flower L1 checkpoints
- P8 QGA/FedAvg Flower checkpoints when generated locally
- P9 QIFA/QIFA+QGA checkpoints when generated locally

These artifacts are local and are not staged automatically by P13. The dashboard can attempt test-set evaluation if a compatible checkpoint is present; otherwise it falls back to reported metrics.

## Test Datasets Available

The L1 preprocessed holdout is available:
- `outputs/preprocessed/l1_binary/test_scaled.npz`
- `outputs/preprocessed/l1_binary/test.parquet`

The L1 train/validation NPZ files also exist locally, but P13 uses only the global test holdout for evaluation. P13 never trains and never uses the test set to choose a model or feature mask.

## QGA Mask Available

The final calibrated P8 mask is available:
- `outputs/qga_feature_selection/final_selected_mask/feature_mask.json`
- `outputs/qga_feature_selection/final_selected_mask/selected_features.json`
- `outputs/qga_feature_selection/final_selected_mask/selection_decision.json`

The selected mask is `conservative_seed_42` with 12 features. It is the required QGA mask for P8 FedAvg + QGA and P9 QIFA + QGA dashboard evaluation.

## Models That Can Be Evaluated

The P13 registry includes:
- P5 FedAvg L1
- P8 FedAvg + QGA L1
- P9 QIFA L1
- P9 QIFA + QGA L1

Each model can be evaluated only if a compatible local checkpoint is found. Otherwise the dashboard marks the model as `report_only` and displays validated P12/P9 metrics.

## Models That May Be Report-Only

P8/P9 checkpoints may be absent or incompatible on another machine because heavy checkpoints are intentionally not committed. In that case:
- P8 FedAvg + QGA remains the recommended production L1 model from P12 evidence.
- P9 QIFA remains the lowest-FPR alternative.
- P9 QIFA + QGA remains the strongest attack-recall/robustness alternative.

## Recommended P13 Architecture

Use a new isolated dashboard under:

`experiments/qi-fl-ids-iot-final/dashboard/`

Recommended stack:
- FastAPI
- local JSON/CSV evidence assets
- optional local checkpoint evaluation
- no Docker requirement
- no change to `services/`

This keeps P13 close to deployment while preserving the existing service dashboard for P14 Docker stack integration.
