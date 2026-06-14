# P13 Dashboard L1 Final

## Objective

P13 provides a final L1 dashboard for the Quantum-Inspired Federated Learning IDS project. It is designed to be close to deployment without starting the P14 Docker stack.

The dashboard:
- displays global P12 results;
- lists the L1 model registry;
- evaluates compatible local checkpoints on the L1 test holdout;
- falls back to validated report metrics when checkpoints are unavailable;
- highlights the recommended production model, P8 FedAvg + QGA;
- displays P10 robustness and P11 compression as complementary panels.

P13 never retrains models and never runs a Flower full training job.

## Modes

### Evidence Mode

Evidence mode reads existing reports:
- `outputs/reports/p12_global_ablation_summary.csv`
- `outputs/reports/p10_robustness_full_summary.csv`
- `outputs/reports/p11_fedtn_mps_summary.csv`

This mode is always available.

### Evaluation Mode

Evaluation mode attempts to resolve model checkpoints from `dashboard/model_registry.json` and evaluate them on:

`outputs/preprocessed/l1_binary/test_scaled.npz`

The test holdout is used only for final evaluation. It is not used for training, model selection, or QGA mask selection.

If a checkpoint is missing or incompatible, the dashboard marks the model `report_only`.

### Demo Mode

Demo/replay mode is reserved for P14. P13 does not integrate live service traffic or Docker orchestration.

## Included Models

The dashboard registry includes:
- P5 FedAvg L1
- P8 FedAvg + QGA L1
- P9 QIFA L1
- P9 QIFA + QGA L1

Recommended deployment model:

`P8 FedAvg + QGA L1`

Reason:
- strong P12 Macro-F1 and attack recall;
- 12 selected features from calibrated QGA mask `conservative_seed_42`;
- lower bandwidth than full-feature FedAvg;
- true Flower runtime evidence.

Alternatives:
- P9 QIFA L1: lowest-FPR alternative.
- P9 QIFA + QGA L1: strongest attack recall and robustness alternative.

## Commands

Build dashboard assets:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/13_build_dashboard_assets.py
```

Evaluate dashboard models:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/13_evaluate_dashboard_models.py
```

Run the dashboard locally:

```powershell
cd experiments/qi-fl-ids-iot-final/dashboard
python app.py
```

Open:

```text
http://127.0.0.1:8013
```

## Outputs

Dashboard assets:
- `dashboard/data/dashboard_summary.json`
- `dashboard/model_registry.json`

Evaluation reports:
- `outputs/reports/p13_dashboard_model_evaluation.csv`
- `outputs/reports/p13_dashboard_model_evaluation.json`
- `outputs/reports/p13_dashboard_model_evaluation_table.md`

Warnings:
- `outputs/reports/p13_dashboard_warnings.json`

## Limitations

- Heavy checkpoints are not committed, so another machine may show report-only rows.
- P11 FedTN/MPS rank 8 is structural/dry-run only and must not be presented as measured predictive performance.
- P13 does not modify `services/dashboard` or `services/docker-compose.yml`.

## P14 Transition

P14 should package the selected L1 model, QGA mask, scaler, and dashboard/API service into the Docker delivery stack. P13 intentionally keeps the dashboard local and evidence-driven so P14 can make packaging decisions cleanly.
