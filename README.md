# Federated Learning for IoT Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)](#reproducibility)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Final-year project delivery for a distributed intrusion detection system based on Federated Learning (FL) over the CICIoT2023 dataset. The repository combines a centralized baseline, a reference FL implementation, and a final FL experiment stack designed for reproducible evaluation, report generation, and MLflow tracking.

## Project Overview

This project studies how to detect IoT network intrusions without centralizing raw traffic data. Instead of sending local samples to a single trainer, each client trains a local model and only shares model updates with the FL server. The goal is to build a distributed IDS pipeline that remains practical under heterogeneous data distributions.

The final experimental system is centered on `experiments/fl-iot-ids-v3`, with:

- a Flower-based FL server,
- three logical clients: `node1`, `node2`, `node3`,
- the CICIoT2023 dataset as the reference benchmark,
- reproducible scenario-driven experiments,
- MLflow tracking and lightweight result exports.

## System Architecture

The training setup follows a classic cross-silo FL design:

```text
                 +---------------------------+
                 |       FL Server           |
                 |  Round orchestration      |
                 |  Aggregation (FedAvg)     |
                 |  Tracking / MLflow        |
                 +------------+--------------+
                              |
          -----------------------------------------------
          |                      |                      |
    +-----+-----+          +-----+-----+          +-----+-----+
    |   node1   |          |   node2   |          |   node3   |
    | local fit |          | local fit |          | local fit |
    | local eval|          | local eval|          | local eval|
    +-----------+          +-----------+          +-----------+
```

In simple terms:

- the server sends the current global model to all clients,
- each client trains locally on its own partition,
- the server aggregates the returned weights,
- the process repeats for a fixed number of rounds.

The main FL strategies explored in the project are:

- `FedAvg`: the baseline aggregation method,
- `FedProx`: a regularized alternative that helps when client distributions drift apart,
- `SCAFFOLD`: investigated during experimentation, but considered unstable in the final project conclusions.

## Scenarios

The project evaluates three heterogeneous data scenarios:

- `normal_noniid`: clients see different class proportions but keep a broad local distribution.
- `absent_local`: some classes are intentionally missing on some clients to simulate stronger drift.
- `rare_expert`: rare attack knowledge is concentrated on one client, which exposes bias and specialization issues.

These scenarios are used to assess how robust each FL strategy remains when client data is not identically distributed.

## Pipeline Description

The final workflow is:

1. Data preprocessing
   - start from the fixed CICIoT2023 export,
   - apply the selected preprocessing and label mapping,
   - prepare scenario-specific partitions for each node.
2. Scenario generation
   - build the client splits for `node1`, `node2`, and `node3`,
   - preserve the intended non-IID setup for each experiment family.
3. Federated training
   - launch the Flower simulation,
   - run the configured FL strategy and loss function,
   - track metrics and artifacts with MLflow.
4. Evaluation
   - compute final accuracy and macro metrics,
   - export comparison tables and HTML reports for delivery.

## Model

The project uses a compact MLP classifier adapted to the engineered CICIoT2023 feature space:

```text
Input (28 features)
  -> Linear(28, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Dropout(0.2)
  -> Linear(128, 34)
  -> Output logits
```

Training losses used across the experiments include:

- `CrossEntropyLoss`
- `Focal Loss`
- weighted variants for imbalance handling when required by the experiment registry

## FL Strategies

### FedAvg

FedAvg is the main baseline. It performs a weighted average of local client updates and remains the strongest overall reference in the final benchmark.

### FedProx

FedProx introduces a proximal term during local training to reduce client drift. It is especially useful when the scenario becomes more heterogeneous, such as `absent_local`.

### SCAFFOLD

SCAFFOLD was part of the study, but it showed unstable behavior in this project context. It is therefore documented as an explored strategy rather than the default final pipeline.

## Experimental Results

The final lightweight reports are available in `experiments/fl-iot-ids-v3/outputs/reports/`:

- [Experiment comparison HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_experiment_comparison.html)
- [Full ablation table HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_full_ablation_table.html)
- [10-round summary HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_results_10rounds.html)
- [Ablation table CSV](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_ablation_table.csv)
- [Ablation table Markdown](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_ablation_table.md)

For the centralized baseline and additional context, see:

- [Baseline experiment README](experiments/baseline-CIC_IOT_2023/README.md)
- [FL v3 experiment README](experiments/fl-iot-ids-v3/README.md)

## Key Findings

- FedAvg remains a strong baseline across the final benchmark.
- FedProx helps when heterogeneity becomes severe, especially in strongly non-IID settings.
- SCAFFOLD was explored but remained unstable for the retained delivery pipeline.
- The `rare_expert` scenario reveals bias and specialization issues when rare knowledge is concentrated on one client.

## Reproducibility

The repository is prepared for stable project delivery with fixed experimental assumptions:

- random seed: `42`
- dataset path fixed in the project setup: `C:\Users\saifb\dev\qi-fl-ids-iot\data\balancing_v3_fixed300k_outputs`
- final experiment definitions frozen in `experiments/fl-iot-ids-v3/configs/experiment_registry.yaml`
- shared runtime configuration frozen in `experiments/fl-iot-ids-v3/configs/global.yaml`
- heavy artifacts such as datasets, MLflow runs, checkpoints, and serialized models are excluded from Git

For consistent reruns, keep the same configuration files, the same local dataset export, and the same Python environment.

## How to Run

From the repository root:

```powershell
cd experiments\fl-iot-ids-v3
python -m src.scripts.run_experiment --experiment exp_v3_fedavg_normal_classweights
```

Start MLflow UI from the same directory:

```powershell
mlflow ui --backend-store-uri outputs\mlruns
```

Generate the ablation table export:

```powershell
python -m src.scripts.build_ablation_table
```

Useful registry entries include:

- `exp_v3_fedavg_normal_classweights`
- `exp_v3_fedavg_normal_focal`
- `exp_v3_fedavg_normal_focal_weighted`
- `exp_v3_fedprox_absentlocal_classweights`
- `exp_v3_fedavg_rareexpert_focal_weighted`

## Project Structure

```text
qi-fl-ids-iot/
├── data/                           # local datasets and fixed exports (not tracked)
├── docs/                           # project figures and supporting documentation
├── experiments/
│   ├── baseline-CIC_IOT_2023/      # centralized baseline experiments
│   ├── fl-iot-ids-v1/              # initial FL reference version
│   ├── fl-iot-ids-v2/              # donor / intermediate version kept for comparison
│   └── fl-iot-ids-v3/              # final FL system for delivery
│       ├── configs/                # frozen experiment and runtime configs
│       ├── outputs/reports/        # lightweight exported reports
│       └── src/                    # FL code, tracking, scripts, utilities
├── outputs/                        # shared root-level outputs if needed
├── LICENSE
└── README.md
```

## Delivery Notes

- Large local artifacts under `experiments/baseline-CIC_IOT_2023/kaggle/` are intentionally not tracked by Git.
- The repository is intended to keep lightweight reports and code, while datasets, trained models, MLflow runs, and checkpoints remain local.
- The final FL delivery path is `experiments/fl-iot-ids-v3`.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
