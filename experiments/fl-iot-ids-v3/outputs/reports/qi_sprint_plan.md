# QI Optimization Sprint Plan

Date: 2026-05-03

## Phase 1: QIFA

Status: implemented and unit-tested.

- Add `src/fl/qifa_strategy.py`.
- Reuse `ReportingFedAvg` tracking and reporting.
- Add `configs/fl/qifa.yaml`.
- Add registry entries:
  - `exp_v3_qifa_normal_classweights`
  - `exp_v3_qifa_absentlocal_classweights`
- Log per-round QIFA metrics.
- Prove `lambda_qifa=0` and perturbation disabled equals FedAvg.

## Phase 2: QGA Feature Selection

Status: implemented and smoke-tested.

- Add `src/qi/feature_selection.py`.
- Add `src/scripts/run_qi_feature_selection.py`.
- Add `configs/qi/qga_feature_selection.yaml`.
- Add `configs/model/flat_34_qga_k16.yaml`.
- Produce scenario artifacts under `artifacts/qi_feature_selection/<scenario>/`.
- Apply selected feature indices at dataset loading time for train/val NPZs.

## Phase 3: Benchmark Configuration

Status: configured, full experimental results pending.

Benchmark entries:

- FedAvg all-features: `exp_v3_fedavg_normal_classweights`
- QIFA all-features: `exp_v3_qifa_normal_classweights`
- QGA-selected + FedAvg: `exp_v3_qga_fedavg_normal_classweights`
- QGA-selected + QIFA: `exp_v3_qga_qifa_normal_classweights`

## Validation Policy

- Unit tests are required for QIFA and QGA.
- Smoke QGA can run quickly and may use synthetic fallback only when local NPZ
  artifacts are missing.
- Full FL experiments can be long. If not run, results must be marked pending.
- No fabricated metrics are allowed.
