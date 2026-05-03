# QI Sprint Audit

Date: 2026-05-03

## Scope

Audited the scientific FL layer in `experiments/fl-iot-ids-v3`. Microservices
and dashboard code were intentionally left out of the QI integration path.

## Pipeline Findings

- The current FL pipeline is registry-driven through
  `configs/experiment_registry.yaml` and `src/scripts/run_experiment.py`.
- `run_experiment.py` resolves global, FL, model, data and imbalance YAML files,
  validates required artifacts, then launches Flower simulation with
  `create_server_app` and `create_client_app`.
- Strategies are selected in `src/fl/server_app.py` from
  `experiment.fl_strategy` or `strategy.name`.
- The existing reporting path is concentrated in
  `src/fl/reporting_strategy.py`. `ReportingFedAvg` handles FedAvg/FedProx style
  aggregation, expert weighting, MLflow round metrics and best checkpointing.
- FedProx is a client-side training variant in `src/fl/client_app.py`; server
  aggregation remains FedAvg-compatible.
- SCAFFOLD uses `ReportingScaffold`, extending `ReportingFedAvg` with control
  variate synchronization.

## Feature Findings

- The real v3 feature count is 28, not 33.
- `configs/model/flat_34.yaml` sets `input_dim: 28`.
- `src/common/paths.py` defines `INPUT_DIM = 28`.
- Scenario generation writes `feature_names` into every processed NPZ and also
  saves `artifacts/feature_names_<scenario>.pkl`.
- `src/data/dataset.py` loads `X`, `y` and `feature_names` from NPZ files.
- `MLPClassifier` accepts variable `input_dim`; no fixed 28-feature assumption
  exists inside the model class.

## Integration Points

- QIFA belongs in `src/fl/qifa_strategy.py` and should be selected from
  `src/fl/server_app.py`.
- QGA feature selection belongs in `src/qi/feature_selection.py`.
- Runtime feature reduction belongs in the dataset/dataloader/client path, not
  in raw CSV generation and not in scaler fitting.
- Reduced-feature experiments should use a separate model config such as
  `flat_34_qga_k16`.

## Config Coherence

- FL configs under `configs/fl/` define rounds, client counts, local epochs,
  learning rate and proximal parameters.
- Model configs define input/output dimensions and hidden layers.
- Data configs define scenario names and client counts.
- `client_app.py` infers runtime input dimension from actual NPZ tensors, so
  selected-feature masking must happen before model construction.

## Existing Tests

- Data/model smoke tests exist for datasets, model output shape, preprocessing,
  tracking and preflight checks.
- FL invariants already test missing-class client shape consistency and
  FedAvg/FedProx/SCAFFOLD synthetic differences.
- Multi-tier tests exist and should not be touched by the QI sprint.

## Tests Added

- QIFA output shape and FedAvg equivalence tests.
- QIFA one/two/three-client aggregation tests.
- QIFA perturbation disablement and deterministic seed tests.
- QGA exact-K mask repair, deterministic selection and artifact writing tests.

## Git Hygiene

Heavy artifacts are excluded by `.gitignore`: `.npz`, `.npy`, `.pkl`,
checkpoints, MLflow runs, datasets and output logs. Small Markdown/JSON reports
are safe to version.
