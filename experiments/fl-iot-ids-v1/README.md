# fl-iot-ids-v1 — Classical Federated IDS

> **Experiment:** `fl-iot-ids-v1`  
> **Version:** `v1.1-docker-config-stable`  
> **Purpose:** Classical Federated Learning baseline — FedAvg · 3 nodes · Flower · CIC-IoT-2023  
> **Status:** ✅ Local phase complete · ✅ Docker validated · 🐳 Docker Compose ready

---

## Overview

This experiment implements a **privacy-preserving distributed Intrusion Detection System** using Federated Learning. Raw traffic data never leaves each IoT node — only model weights are exchanged with the central server. The global model is built by aggregation, not data pooling.

`fl-iot-ids-v1` is the **second phase** of the PFE pipeline:

```
baseline-CIC_IOT_2023  →  fl-iot-ids-v1  →  fl-iot-ids-v2 (QI)  →  fl-iot-ids-v3 (MLOps)
    (centralized)           (this repo)        (quantum-inspired)       (production)
```

It establishes a reproducible FL baseline (Macro F1 target ≥ 0.51) before introducing Quantum-Inspired optimization in v2.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │         FL Server            │
                    │   FedAvg · Flower · port 8080│
                    └──────────┬──────────────────┘
                               │  gRPC (weights only)
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────▼──┐     ┌──────▼──┐     ┌──────▼──┐
        │ Node 1  │     │ Node 2  │     │ Node 3  │
        │ IoT GW  │     │ IoT GW  │     │ IoT GW  │
        │ MLP     │     │ MLP     │     │ MLP     │
        │ Client  │     │ Client  │     │ Client  │
        └─────────┘     └─────────┘     └─────────┘

  Local data stays on each node — only Δweights are shared.
```

**Model:** MLP (PyTorch) — `input(33) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(34) → Softmax`  
**Strategy:** FedAvg — weighted average by `n_samples` per node  
**Partitioning:** Dirichlet (α=0.5) — non-IID to simulate realistic IoT deployments  
**Dataset:** CIC-IoT-2023 — 34 attack classes · 33 features · ~5.7M training samples

---

## Results

### FL Convergence (3 rounds, Docker validation)

| Round | Distributed Loss | Client Accuracy (avg) |
|---|---|---|
| 1 | 0.7793 | ~0.74 |
| 2 | 0.4157 | ~0.86 |
| 3 | 0.3303 | ~0.90 |

### Per-client Round 3

| Client | Train Loss | Train Acc | Eval Loss | Eval Acc |
|---|---|---|---|---|
| node1 | 0.4216 | 0.8851 | 0.3310 | 0.8990 |
| node2 | 0.4463 | 0.8824 | 0.3313 | 0.8998 |
| node3 | 0.4603 | 0.8805 | 0.3269 | 0.9003 |

> Macro F1 per-round will be reported after Docker Compose end-to-end validation with evaluation set.

### Smoke Test Summary

| Criterion | Status |
|---|---|
| Docker build (base / server / client) | ✅ PASS |
| Server startup | ✅ PASS |
| 3 clients connected | ✅ PASS |
| 3 rounds — 0 failures | ✅ PASS |
| `aggregate_fit` received 3/3 results | ✅ PASS |
| `aggregate_evaluate` received 3/3 results | ✅ PASS |
| Loss decreasing across rounds | ✅ PASS |
| Accuracy increasing across rounds | ✅ PASS |

---

## Repository Structure

```
fl-iot-ids-v1/
├── src/                        ← Python source code (see detail below)
│   ├── common/                 ← Config, logging, paths, schemas, utils
│   ├── data/                   ← Data pipeline (partitioning, preprocessing, dataloader, dataset)
│   ├── fl/                     ← Flower FL layer (client_app, server_app, strategy, metrics)
│   ├── model/                  ← MLP network, training loop, evaluation, losses
│   ├── scripts/                ← Executable entry points (run_server, run_client, prepare_partitions, ...)
│   └── services/               ← Service layer (fl_client_service, preprocessor_service, collector_service)
├── configs/                    ← YAML configuration files
│   ├── global.yaml             ← Global project settings
│   ├── fl_config.yaml          ← FL hyperparameters (rounds, epochs, lr, batch size)
│   ├── model.yaml              ← MLP architecture definition
│   └── nodes/                  ← Per-node overrides
│       ├── node1.yaml
│       ├── node2.yaml
│       └── node3.yaml
├── deployments/docker/         ← Docker build files and Compose
│   ├── base.Dockerfile         ← Shared CPU-only base image (~3.28GB)
│   ├── server.Dockerfile       ← Extends base — FL server entrypoint
│   ├── client.Dockerfile       ← Extends base — FL client entrypoint
│   └── docker-compose.yml      ← Orchestrates server + 3 clients
├── artifacts/                  ← Preprocessing artifacts from baseline (never committed)
│   ├── scaler_robust.pkl       ← RobustScaler fitted on full CIC-IoT-2023 train
│   ├── label_mapping_34.pkl    ← 34-class label ↔ int mapping
│   ├── class_weights_34.pkl    ← Balanced class weights for CrossEntropyLoss
│   ├── feature_names.pkl       ← Ordered list of 33 feature names
│   └── baseline/               ← Original copies from baseline-CIC_IOT_2023
├── data/                       ← Node-local data (never committed)
│   ├── raw/node{1,2,3}/train.csv          ← Dirichlet-partitioned CSVs
│   └── processed/node{1,2,3}/train_preprocessed.npz  ← Scaled + encoded arrays
├── outputs/                    ← Runtime outputs (never committed)
│   ├── logs/                   ← fl_server.log, fl_client.log, run_*.log
│   ├── checkpoints/            ← Saved model weights per round
│   ├── metrics/                ← Per-round metric JSON files
│   └── reports/                ← Evaluation reports
├── tests/                      ← Unit + smoke tests
│   ├── test_dataset.py
│   ├── test_fl_smoke.py
│   ├── test_model.py
│   └── test_preprocessor.py
├── docs/                       ← Engineering documentation
│   ├── architecture.md         ← Full system architecture document
│   ├── local_v1_runbook.md     ← Step-by-step local execution guide
│   ├── local_v1_acceptance.md  ← Functional acceptance criteria (all passed)
│   ├── validation_smoke_test_v1.md  ← Docker smoke test report
│   └── release_v1_1_docker_config_stable.md  ← Release note for v1.1
├── requirements.txt
├── requirements-lock.txt       ← Frozen environment for reproducibility
├── pyproject.toml
└── VERSION
```

---

## Source Code — `/src` Detail

### `src/common/`

Shared utilities used across all modules.

| File | Role |
|---|---|
| `config.py` | YAML config loader — merges `global.yaml`, `fl_config.yaml`, `model.yaml`, and the active node config |
| `logger.py` | Structured logger factory — writes to `outputs/logs/` with rotation |
| `paths.py` | Centralized path resolver — all `data/`, `artifacts/`, `outputs/` paths defined here |
| `schemas.py` | Pydantic schemas for config validation (FL config, node config, model config) |
| `utils.py` | Seed fixing, tensor conversion helpers, metric formatting |

### `src/data/`

Full data pipeline from raw CSV to PyTorch `DataLoader`.

| File | Role |
|---|---|
| `partitioning.py` | Dirichlet partitioning (α=0.5) of CIC-IoT-2023 into 3 non-IID node splits — saves `data/raw/node{i}/train.csv` |
| `preprocessor.py` | Applies baseline-compatible preprocessing: loads `scaler_robust.pkl`, `label_mapping_34.pkl`, `feature_names.pkl` — saves `train_preprocessed.npz` |
| `dataset.py` | `torch.utils.data.Dataset` wrapper over `.npz` files — returns `(X_tensor, y_tensor)` pairs |
| `dataloader.py` | Builds `DataLoader` with `batch_size=256`, `shuffle=True`, `num_workers=0` (Docker-safe) |
| `collector.py` | Optional: network traffic collector interface (placeholder for v2 edge deployment) |

### `src/fl/`

Flower FL integration layer.

| File | Role |
|---|---|
| `client_app.py` | `FlowerClient` — implements `get_parameters()`, `set_parameters()`, `fit()`, `evaluate()`. Wraps local training and evaluation. |
| `server_app.py` | `ServerApp` — configures Flower server with `FedAvgIDS` strategy, round count, minimum clients |
| `strategy.py` | `FedAvgIDS` — extends `FedAvg` with per-round metric logging (loss, accuracy, Benign Recall). Custom `aggregate_fit` and `aggregate_evaluate`. |
| `metrics.py` | Metric computation helpers: Macro F1, per-class recall, Benign Recall — used inside `evaluate()` |

### `src/model/`

PyTorch MLP model definition and training loop.

| File | Role |
|---|---|
| `network.py` | `MLPClassifier` — `nn.Module` with configurable hidden layers, ReLU activations, final softmax. Input: 33 features. Output: 34 classes. |
| `train.py` | `train_one_epoch()` — one epoch of Adam + CrossEntropyLoss with class weights. Returns `(loss, accuracy)`. |
| `evaluate.py` | `evaluate_model()` — computes accuracy, Macro F1, and per-class recall on a DataLoader |
| `losses.py` | `WeightedCrossEntropyLoss` — wraps `nn.CrossEntropyLoss` with `class_weights_34.pkl` tensor |

### `src/scripts/`

All executable entry points. Run with `python -m src.scripts.<name>`.

| Script | Command | Role |
|---|---|---|
| `prepare_partitions.py` | `python -m src.scripts.prepare_partitions` | Reads baseline CSV, applies Dirichlet split, writes `data/raw/node{i}/train.csv` + `partition_manifest.json` |
| `preprocess_node_data.py` | `python -m src.scripts.preprocess_node_data --node-id node1` | Preprocesses one node's raw CSV using baseline artifacts → `data/processed/node{i}/train_preprocessed.npz` |
| `run_server.py` | `python -m src.scripts.run_server --host 0.0.0.0 --port 8080 --num-rounds 3 --min-clients 3` | Starts Flower FL server |
| `run_client.py` | `python -m src.scripts.run_client --node-id node1 --server-address 127.0.0.1:8080 --local-epochs 1` | Starts one Flower client for a given node |
| `test_dataloader.py` | `python -m src.scripts.test_dataloader` | Smoke test: loads node1 DataLoader, prints batch shape |
| `test_local_training.py` | `python -m src.scripts.test_local_training` | Smoke test: one epoch of local training on node1, checks loss decreasing |
| `smoke_test.py` | `python -m src.scripts.smoke_test` | Full pipeline smoke test without Flower |

### `src/services/`

Service layer — thin orchestration wrappers for deployment contexts.

| File | Role |
|---|---|
| `fl_client_service.py` | Service wrapper around `FlowerClient` — handles startup, config injection, error recovery |
| `preprocessor_service.py` | Service wrapper around `Preprocessor` — used by Docker entrypoint |
| `collector_service.py` | Placeholder for edge traffic collection service (v2) |

---

## Prerequisites

### Required artifacts (from `baseline-CIC_IOT_2023`)

Before running any step, copy the 4 preprocessing artifacts:

```bash
cp ../baseline-CIC_IOT_2023/artifacts/scaler_robust.pkl    artifacts/
cp ../baseline-CIC_IOT_2023/artifacts/label_mapping_34.pkl  artifacts/
cp ../baseline-CIC_IOT_2023/artifacts/class_weights_34.pkl  artifacts/
cp ../baseline-CIC_IOT_2023/artifacts/feature_names.pkl     artifacts/
```

These are already present in `artifacts/` if you followed the baseline pipeline.

### Environment setup

```bash
cd experiments/fl-iot-ids-v1

conda activate fl-iot-ids
pip install -r requirements.txt
```

---

## Quick Start — Local (without Docker)

### Step 1 — Prepare data partitions

```bash
# Windows PowerShell
$env:PYTHONPATH = "."
python -m src.scripts.prepare_partitions
```

**Output:**
```
data/raw/node1/train.csv
data/raw/node2/train.csv
data/raw/node3/train.csv
data/splits/partition_manifest.json
```

### Step 2 — Preprocess each node

```bash
python -m src.scripts.preprocess_node_data --node-id node1
python -m src.scripts.preprocess_node_data --node-id node2
python -m src.scripts.preprocess_node_data --node-id node3
```

**Output:** `data/processed/node{i}/train_preprocessed.npz`

### Step 3 — Verify (optional smoke tests)

```bash
python -m src.scripts.test_dataloader       # check batch shape (256, 33)
python -m src.scripts.test_local_training   # check loss decreasing over 1 epoch
```

### Step 4 — Run FL (4 terminals)

```bash
# Terminal 1 — server
python -m src.scripts.run_server --host 127.0.0.1 --port 8080 --num-rounds 3 --min-clients 3

# Terminal 2
python -m src.scripts.run_client --node-id node1 --server-address 127.0.0.1:8080 --local-epochs 1

# Terminal 3
python -m src.scripts.run_client --node-id node2 --server-address 127.0.0.1:8080 --local-epochs 1

# Terminal 4
python -m src.scripts.run_client --node-id node3 --server-address 127.0.0.1:8080 --local-epochs 1
```

**Expected output (server):**
```
aggregate_fit: received 3 results and 0 failures
aggregate_evaluate: received 3 results and 0 failures
History (loss, distributed): round 1: ~0.78 | round 2: ~0.42 | round 3: ~0.33
```

---

## Quick Start — Docker

### Step 1 — Build images

```bash
# Base image (do this first — ~3.28GB, takes a few minutes)
docker build -f deployments/docker/base.Dockerfile -t fl-iot-ids-v1:latest .

# Server and client images
docker build -f deployments/docker/server.Dockerfile -t fl-iot-server:v1 .
docker build -f deployments/docker/client.Dockerfile -t fl-iot-client:v1 .
```

### Step 2 — Validate images

```bash
docker run --rm fl-iot-ids-v1:latest python -c "import torch; import flwr; print('OK')"
docker run --rm fl-iot-server:v1 python -c "print('server image OK')"
docker run --rm fl-iot-client:v1 python -c "print('client image OK')"
```

### Step 3 — Run with Docker Compose

```bash
docker compose -f deployments/docker/docker-compose.yml up
```

This starts: 1 server + 3 clients. Data volumes are mounted from `data/processed/` and `artifacts/`.

---

## Configuration

All FL hyperparameters are in `configs/fl_config.yaml`:

```yaml
# Key parameters
num_rounds: 3
min_available_clients: 3
local_epochs: 1
batch_size: 256
learning_rate: 0.001
server_address: "0.0.0.0:8080"
```

Per-node overrides are in `configs/nodes/node{i}.yaml` — useful for simulating heterogeneous IoT hardware constraints (different batch sizes, learning rates).

---

## Feature Alignment

A critical requirement for federated IDS: every node must produce **identical feature vectors** regardless of which local CSV partition it received.

This is guaranteed by the **shared artifacts** in `artifacts/`:

```
artifacts/
  ├── scaler_robust.pkl      ← Same RobustScaler transform on every node
  ├── feature_names.pkl      ← Same 33 features in the same order
  ├── label_mapping_34.pkl   ← Same integer encoding for all 34 classes
  └── class_weights_34.pkl   ← Same loss weights across all nodes
```

In Docker, these are mounted as a **read-only volume** shared by all containers. No node ever refits the scaler independently — this would introduce distribution shift.

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific suites
pytest tests/test_dataset.py -v          # Dataset loading
pytest tests/test_model.py -v            # MLP forward pass, parameter count
pytest tests/test_preprocessor.py -v    # Artifact loading, feature alignment
pytest tests/test_fl_smoke.py -v        # FL round simulation (no network)

# Skip slow tests
pytest tests/ -v -m "not slow"
```

---

## Logs

All logs are written to `outputs/logs/`:

| File | Content |
|---|---|
| `fl_server.log` | Round events, aggregation results, strategy decisions |
| `fl_client.log` | Per-round training loss/accuracy, evaluation results |
| `run_server.log` | Server startup, gRPC connection events |
| `run_client.log` | Client startup, connection to server, Flower handshake |

---

## Known Limitations (V1)

These are known and accepted at this stage:

| Limitation | Impact | Planned fix |
|---|---|---|
| `start_server()` / `start_client()` deprecated | Warning only — no functional impact | Migrate to `flower-superlink` in v2 |
| `No fit_metrics_aggregation_fn provided` | Metrics not globally aggregated | Add in strategy.py |
| Ray warnings on Windows | Log noise only | Use Linux/WSL2 for production |
| Local eval uses local train partition | No global test set evaluation per round | Add centralized eval dataset in v2 |
| No MLflow / Grafana | No experiment tracking UI | Planned for v3 (MLOps) |

---

## Connection to Other Experiments

### Inputs (from `baseline-CIC_IOT_2023`)

```
baseline-CIC_IOT_2023/artifacts/scaler_robust.pkl    →  artifacts/scaler_robust.pkl
baseline-CIC_IOT_2023/artifacts/label_mapping_34.pkl →  artifacts/label_mapping_34.pkl
baseline-CIC_IOT_2023/artifacts/class_weights_34.pkl →  artifacts/class_weights_34.pkl
baseline-CIC_IOT_2023/artifacts/feature_names.pkl    →  artifacts/feature_names.pkl
```

### Comparison target (from `baseline-CIC_IOT_2023/results_baseline/`)

| Metric | Centralized baseline | FL v1 target |
|---|---|---|
| Accuracy | 0.9518 | ≥ 0.92 |
| Macro F1 | 0.5106 | ≥ 0.48 |
| BenignTraffic Recall | 0.501 | ↑ (primary target) |

### Outputs (for `fl-iot-ids-v2`)

```
fl-iot-ids-v1/
  ├── src/model/network.py   →  MLP architecture reused in v2 (QI extensions applied on top)
  ├── src/fl/strategy.py     →  FedAvg baseline replaced by QI-FedAvg in v2
  └── configs/fl_config.yaml →  Hyperparameters inherited as starting point for v2
```

---

## Release History

| Tag | Description |
|---|---|
| `v1.0.0` | Initial commit — local FL phase complete |
| `v1.1-docker-config-stable` | Docker images validated · Docker Compose orchestration ready · smoke test passed |

---

## Documentation Index

| Document | Location | Description |
|---|---|---|
| Architecture | `docs/architecture.md` | Full system design — data pipeline, FL protocol, QI layer, Docker stack |
| Local runbook | `docs/local_v1_runbook.md` | Step-by-step local execution with troubleshooting |
| Acceptance | `docs/local_v1_acceptance.md` | Functional acceptance criteria — all 18 criteria passed |
| Smoke test | `docs/validation_smoke_test_v1.md` | Docker smoke test report with full logs |
| Release note | `docs/release_v1_1_docker_config_stable.md` | v1.1 release scope and frozen elements |