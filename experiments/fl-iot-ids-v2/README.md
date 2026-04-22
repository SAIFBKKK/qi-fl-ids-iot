# FL IoT IDS v2 — Robust Federated Learning Benchmark

[![Experiment Status](https://img.shields.io/badge/Status-Active%20Benchmark-orange.svg)](#current-status)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-Modern%20App%20Model-FF6B6B.svg)](https://flower.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
[![Phase](https://img.shields.io/badge/Phase-Robust%20FL%20Benchmark-purple.svg)](#experimental-roadmap)

**Experiment Family:** `fl-iot-ids-v2`  
**Phase:** Robust Federated Learning Benchmark  
**Status:** Baseline validated, comparative benchmarking in progress

---

## Overview

`fl-iot-ids-v2` is the second experimental phase of the FL-based IoT IDS project.
It extends the initial classical federated baseline by introducing a more rigorous framework for studying:

- non-IID federated learning
- class imbalance handling
- rare attack detection
- benign traffic preservation
- communication-aware FL evaluation

This version is designed not only to run FL experiments, but also to produce scientifically comparable results across multiple strategies, scenarios, and model settings.

---

## Project Positioning

```text
baseline-CIC_IOT_2023      fl-iot-ids-v1                 fl-iot-ids-v2                   fl-iot-ids-v3
   (Centralized ML)     (Classical FL Baseline)     (Robust FL Benchmark)           (Future Production/MLOps)
      Reference              Initial validation       Comparative experimentation      Deployment & optimization
```

### Evolution from v1 to v2

Compared to `fl-iot-ids-v1`, this version introduces:

- modern Flower application structure
- experiment registry and config-driven execution
- multiple FL algorithms:
  - FedAvg
  - FedProx
  - SCAFFOLD
- multiple data scenarios:
  - `normal_noniid`
  - `absent_local`
  - `rare_expert`
- imbalance handling strategies:
  - `none`
  - `class_weights`
  - `focal_loss`
  - `light_oversampling`
- IDS-oriented evaluation metrics:
  - macro-F1
  - benign recall
  - false positive rate
  - rare-class recall
  - communication overhead
  - inter-round stability

---

## Scientific Objective

The goal of `fl-iot-ids-v2` is to benchmark how different FL strategies behave under realistic IoT constraints, especially when:

- client data is strongly non-IID
- attack families are imbalanced
- some rare attacks are hard to learn
- preserving benign traffic recognition is critical

This phase is used to answer questions such as:

- Does FedProx improve robustness under non-IID data?
- Do class weights improve rare attack recall?
- What is the trade-off between rare attack detection and false positives?
- Which FL configuration offers the best IDS-oriented compromise?

---

## Current Status

The v2 framework is already functional and experimentally validated at the system level:

- scenario-aware partitioning is implemented
- local preprocessing per scenario/client is operational
- Flower server/client communication is validated
- official FedAvg baseline has been executed successfully
- FedProx with class weights has been integrated and tested
- server-side metrics aggregation is active
- reproducible experiment artifacts are exported automatically

---

## Validated Experiments

### Official baseline

**Experiment:** `exp_flat_fedavg_normal_none`

| Metric | Value |
|--------|-------|
| Accuracy | 0.424 |
| Macro-F1 | 0.295 |
| Recall Macro | 0.391 |
| Benign Recall | 0.161 |
| False Positive Rate | 0.839 |
| Rare Class Recall | 0.024 |

### Additional validated experiments

| Experiment | Rounds | Accuracy | Macro-F1 | Benign Recall | FPR ↓ | Rare Recall |
|-----------|--------|----------|----------|---------------|-------|-------------|
| `exp_flat_fedavg_normal_none` | 10 | 0.424 | 0.295 | 0.161 | 0.839 | 0.024 |
| `exp_flat_fedavg_normal_classweights_v1style` | 10 | 0.423 | 0.308 | 0.168 | 0.832 | 0.103 |
| `exp_flat_fedprox_normal_classweights` | 3 | 0.420 | 0.312 | 0.138 | 0.862 | 0.147 |
| `exp_flat_fedprox_normal_classweights_v1style` | 3 | 0.417 | 0.303 | 0.008 | 0.992 | 0.156 |

---

## Best Results So Far

### Best overall compromise

**Current best candidate:** `FedAvg + class_weights (v1-style hyperparameters)`

Why this candidate stands out:

- best compromise between macro-F1 and benign traffic preservation
- clearly better rare-class recall than the plain FedAvg baseline
- lower false positive rate than the FedProx variants tested so far
- more balanced behavior for a practical IDS setting

### Key experimental finding

The benchmark already highlights a strong trade-off:

- **FedProx + class_weights** improves rare attack recall
- but may over-predict attacks and degrade benign recall
- **FedAvg + class_weights** currently provides the most balanced IDS-oriented behavior

---

## Comparative Interpretation

### FedAvg baseline (`none`)
- stable baseline
- weak rare attack recall
- moderate benign recognition

### FedAvg + class_weights (v1-style)
- improves macro-F1
- improves rare attack recall substantially
- preserves the best benign/FPR balance among tested variants

### FedProx + class_weights
- strongest rare-class recall among practical candidates
- promising for rare attack sensitivity
- still too aggressive on benign traffic

### FedProx + class_weights (v1-style)
- best rare recall among tested runs
- but severe benign collapse and extremely high FPR
- not suitable as the current main candidate

---

## Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                 FL Server / Strategy Layer                 │
│     FedAvg / FedProx / SCAFFOLD + metrics aggregation      │
└───────────────┬───────────────────┬────────────────────────┘
                │                   │
         ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
         │   Client 1   │     │   Client 2   │     │   Client 3   │
         │  non-IID     │     │  non-IID     │     │ rare expert  │
         │ local train  │     │ local train  │     │ optional     │
         └──────────────┘     └──────────────┘     └──────────────┘

Privacy guarantee: raw data remains local
Communication: model parameters / updates only
Evaluation: federated metrics aggregated at server side
```

---

## Benchmark Dimensions

### A. FL Algorithm
- `fedavg`
- `fedprox`
- `scaffold`

### B. Model Architecture
- `flat_34`
- `hierarchical`

### C. Data Scenario
- `normal_noniid`
- `absent_local`
- `rare_expert`

### D. Imbalance Handling
- `none`
- `class_weights`
- `focal_loss`
- `light_oversampling`

---

## Metrics

This project evaluates FL-IDS quality using metrics aligned with real intrusion detection constraints:

- accuracy
- macro-F1
- recall macro
- benign recall
- false positive rate
- rare-class recall
- update size (bytes)
- training/evaluation time
- inter-round convergence behavior

Unlike the first baseline phase, v2 explicitly measures the trade-off between:

- detecting rare attacks
- preserving benign traffic
- reducing false alarms

---

## Repository Layout

```text
fl-iot-ids-v2/
├── configs/        # experiment, FL, model, data, imbalance, node configs
├── src/            # source code
├── data/           # raw and processed scenario-specific data
├── artifacts/      # shared and scenario artifacts
├── outputs/        # reports, metrics, logs, checkpoints, figures
├── mlruns/         # MLflow local tracking
├── tests/          # unit and integration tests
├── docs/           # engineering and experiment documentation
└── deployments/    # Docker and MLflow deployment files
```

---

## How to Reproduce the Official Baseline

### 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

### 2. Run the official baseline

```bash
python -m src.scripts.run_experiment --experiment exp_flat_fedavg_normal_none
```

### 3. Outputs

Artifacts are exported automatically under:

```text
outputs/reports/baselines/exp_flat_fedavg_normal_none/
```

Typical files:
- `run_summary.json`
- `round_metrics.json`
- `baseline_notes.md`

---

## Example Experiments

```bash
python -m src.scripts.run_experiment --experiment exp_flat_fedavg_normal_none
python -m src.scripts.run_experiment --experiment exp_flat_fedavg_normal_classweights_v1style
python -m src.scripts.run_experiment --experiment exp_flat_fedprox_normal_classweights
python -m src.scripts.run_experiment --experiment exp_flat_fedprox_normal_classweights_v1style
```

---

## Design Principles

- config-driven experimentation
- strict separation between scenarios and outputs
- reproducible seeded runs
- scenario-aware preprocessing
- server-side aggregation of IDS metrics
- benchmark-first design before production optimization

---

## Execution Model

This project follows the modern Flower application style:

- `ServerApp`
- `ClientApp`
- strategy-driven orchestration
- simulation-based benchmarking
- federated evaluation with aggregated metrics

---

## Experimental Roadmap

### Ablation A — Architecture
- `flat_34`
- `hierarchical`

### Ablation B — FL Algorithm
- `fedavg`
- `fedprox`
- `scaffold`

### Ablation C — Rare Data Scenario
- without expert client
- with expert client

### Ablation D — Imbalance Handling
- `none`
- `class_weights`
- `focal_loss`
- `light_oversampling`

---

## Next Steps

Short-term priorities:

- best-round selection and checkpoint selection
- false positive reduction
- threshold tuning
- calibration experiments
- SCAFFOLD benchmarking
- expert-client evaluation

Mid-term priorities:

- hierarchical evaluation pipeline
- MLflow integration completion
- broader ablation automation
- publication-ready comparative tables and figures

Long-term priorities:

- quantum-inspired weighting and orchestration
- production-oriented deployment
- MLOps and edge execution support

---

## Notes

This repository is under active experimental development.
The codebase is already suitable for comparative FL benchmarking, but results should always be interpreted together with the selected:

- FL strategy
- imbalance handling method
- scenario
- stopping round
- evaluation metric
