# CLAUDE.md — QI-Lab Codebase Guide

## Project Overview

Quantum-Inspired Federated Learning for IoT Intrusion Detection System.
Active version: **`experiments/fl-iot-ids-v2/`** (v2.1.0-stable).
Defense deadline: **June 1, 2026** — Military Academy, Networks & AI (PFE).

---

## Repo Layout

```
qi-fl-ids-iot/
├── experiments/
│   ├── fl-iot-ids-v2/          ← ACTIVE codebase (work here)
│   ├── fl-iot-ids-v1/          ← archived v1, do not modify
│   ├── baseline-CIC_IOT_2023/  ← centralized baseline for comparison
│   └── fl-iot-ids-v3/          ← future stub
├── data/                        ← external raw datasets (never commit)
├── outputs/                     ← MLflow tracking root
├── docs/
├── .github/workflows/ci.yml
├── environment.yml              ← conda env (primary)
└── CLAUDE.md                    ← this file
```

---

## Active Codebase — `experiments/fl-iot-ids-v2/`

### Source layout

```
src/
├── common/         config, logger, paths, registry, seeds, utils
├── data/           partitioning (Dirichlet/absent_local/rare_expert),
│                   preprocessing, balancing, datasets, analysis
├── fl/
│   ├── client/     BaseIDSClient (FedAvg + FedProx), StandardClient, ExpertClient
│   ├── server/     ServerApp factory, strategy_factory, FedAvg/FedProx/SCAFFOLD strategies
│   ├── simulation/ client_factory (make_client_fn), runner, experiment_plan
│   └── metrics/    classification, rare_attack, convergence, stability
├── models/
│   ├── flat/       FlatMLP (34-class), train, evaluate
│   └── hierarchical/ level1_binary, level2_family, level3_specialists, pipeline
├── scripts/        run_experiment.py (main), run_server.py*, run_client.py*,
│                   prepare_partitions.py, preprocess_node_data.py,
│                   build_client_reports.py, validate_partition_scenario.py,
│                   launch_mlflow.py, run_ablation_suite.py
└── tracking/       artifact_logger.py, mlflow_logger.py, run_naming.py
                    (* = distributed mode, planned for v3)
```

### Running an experiment

```bash
cd experiments/fl-iot-ids-v2
python -m src.scripts.run_experiment --experiment exp_flat_fedprox_normal_classweights
```

### Via Docker

```bash
cd experiments/fl-iot-ids-v2
docker compose build
docker compose run --rm fl-simulation --experiment exp_flat_fedprox_normal_classweights
docker compose up mlflow   # MLflow UI at http://localhost:5000
```

### Registered experiments

Defined in `configs/experiment_registry.yaml`. Each experiment composes:

| Key | Source |
|-----|--------|
| `fl_config` (or `fl_strategy`) | `configs/fl/<name>.yaml` |
| `architecture` | `configs/model/<name>.yaml` |
| `data_scenario` | `configs/data/<name>.yaml` |
| `imbalance_strategy` | `configs/imbalance/<name>.yaml` |

Current experiments:
- `exp_flat_fedavg_normal_none` — FedAvg baseline, no imbalance handling
- `exp_flat_fedprox_normal_classweights` — **main baseline** (FedProx + class weights)
- `exp_flat_scaffold_absentlocal_classweights` — SCAFFOLD on absent-local scenario
- `exp_hierarchical_fedprox_rareexpert_classweights` — hierarchical + rare-expert node
- `exp_flat_fedprox_normal_classweights_smoke` — quick 3-round smoke test
- `exp_flat_fedavg_normal_classweights_v1style` — v1 hyperparams for comparison
- `exp_flat_fedprox_normal_classweights_v1style` — v1 hyperparams for comparison

### Non-IID data scenarios

| Scenario | Description |
|----------|-------------|
| `normal_noniid` | Dirichlet allocation across 3 nodes |
| `absent_local` | Some attack classes missing per node |
| `rare_expert` | Node 3 specialises in rare attack classes |

Processed data lives in `data/processed/<scenario>/node{1,2,3}/`.
Never commit `*.npz`, `*.csv`, or any file in `data/`.

### Key classes

- **`BaseIDSClient`** (`src/fl/client/base_client.py`) — single client for both FedAvg and FedProx. FedProx proximal term enabled via `fl_strategy="fedprox"` + `proximal_mu > 0`.
- **`BaselineArtifactTracker`** (`src/tracking/artifact_logger.py`) — thread-safe per-round metric collector; writes JSON + Markdown reports to `outputs/reports/baselines/<exp_name>/`.
- **`MLflowRunLogger`** (`src/tracking/mlflow_logger.py`) — MLflow layer on top of artifact tracker; call `start()` before simulation, `finish()` in `finally`.
- **`make_client_fn`** (`src/fl/simulation/client_factory.py`) — factory consumed by Flower's `ClientApp`; resolves partition path and selects `StandardClient` or `ExpertClient`.

---

## Development

### Environment setup

```bash
conda env create -f environment.yml
conda activate qi-fl-ids
cd experiments/fl-iot-ids-v2
pip install -e .
```

### Tests

```bash
cd experiments/fl-iot-ids-v2
pytest tests/ -v -m "not slow"
```

### Linting

```bash
ruff check src/
```

### CI/CD

GitHub Actions runs on push to `main`/`develop` and PRs to `main`.
Pipeline: lint (ruff) + smoke tests for **both** v1 (legacy) and v2 (active).

---

## What NOT to do

- Never commit `*.npz`, `*.csv`, `*.pkl`, `*.pt`, `*.pth` — all gitignored.
- Never commit `.venv/` or `.venv.zip` — gitignored.
- Never modify `experiments/fl-iot-ids-v1/` — it is a frozen reference.
- Never hardcode absolute paths — always use `src.common.paths`.
- Never push to `main` directly — open a PR from a feature branch.
