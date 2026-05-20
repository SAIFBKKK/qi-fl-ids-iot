# P13 Dashboard L1 Final Plan

## Decision

Create an isolated FastAPI dashboard in:

`experiments/qi-fl-ids-iot-final/dashboard/`

The existing `services/dashboard` app is not modified. It remains reserved for P14 Docker stack integration.

## Modes

### Evidence Mode

Evidence mode reads the validated reports:
- P12 global ablation
- P10 robustness full results
- P11 FedTN/MPS compression dry-run
- P8 calibrated QGA mask

This mode is always available and does not require checkpoints.

### Evaluation Mode

Evaluation mode attempts to load compatible local checkpoints and evaluates models on:

`outputs/preprocessed/l1_binary/test_scaled.npz`

The test holdout is used only for final evaluation. It is never used for model selection, mask selection, training, or threshold tuning.

If a checkpoint is missing or incompatible, the model is marked `report_only` and validated report metrics are displayed.

### Demo / Replay Mode

Demo or replay mode is optional future work for P14. It can be connected later to service-level traffic replay or Dockerized model serving. P13 does not implement live retraining or service orchestration.

## Dashboard Panels

1. Overview
   - Project
   - L1 binary IDS
   - Flower runtime evidence
   - QGA, QIFA, FedTN/MPS status

2. Recommended Deployment Model
   - P8 FedAvg + QGA
   - 12 features
   - `selected_mask_id=conservative_seed_42`
   - Macro-F1, attack recall, FPR, accuracy, bandwidth

3. Test Set Evaluation
   - Recomputed metrics if checkpoint is available
   - Reported metrics otherwise
   - Model status: `evaluable` or `report_only`

4. L1 Comparison
   - P5 FedAvg
   - P8 FedAvg + QGA
   - P9 QIFA
   - P9 QIFA + QGA

5. Robustness
   - P10 label-flip full scenario
   - Best robust method: QIFA + QGA

6. Compression
   - P11 FedTN/MPS rank 8 dry-run
   - Dense vs compressed size
   - Estimated bandwidth reduction
   - Clear dry-run warning

7. Evidence and Reproducibility
   - Report paths
   - Model registry
   - API endpoints
   - Git tags

## Data Sources

- `outputs/reports/p12_global_ablation_summary.csv`
- `outputs/reports/p10_robustness_full_summary.csv`
- `outputs/reports/p11_fedtn_mps_summary.csv`
- `outputs/qga_feature_selection/final_selected_mask/`
- `outputs/preprocessed/l1_binary/test_scaled.npz`
- `dashboard/model_registry.json`

## Endpoints

- `GET /`
- `GET /api/summary`
- `GET /api/models`
- `GET /api/evaluations`
- `GET /api/figures`
- `POST /api/evaluate/{model_id}`
- `GET /health`
- `GET /ready`

## Limits

- P13 does not retrain.
- P13 does not run Flower full.
- P13 does not modify P8/P9/P10/P11/P12 outputs.
- P13 does not touch Docker or the existing service dashboard.
- P11 compression is marked dry-run/structural when no checkpoint-based metric evaluation exists.

## Link With P14

P13 provides the model registry, evidence assets, and a local dashboard layout that can be lifted into P14 Docker stack work. P14 should decide how to package model artifacts, scaler files, and live inference endpoints.
