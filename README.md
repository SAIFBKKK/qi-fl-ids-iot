# Quantum-Inspired Federated Learning for IoT Intrusion Detection

[![CI](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.x-FF6B6B.svg)](https://flower.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/Research-Quantum--Inspired-6f42c1.svg)](#quantum-inspired-roadmap)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)](#reproducibility)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Final-year engineering project for a distributed IoT Intrusion Detection System (IDS) based on Federated Learning (FL), with a quantum-inspired optimization roadmap. The repository combines a centralized CICIoT2023 baseline, earlier FL iterations, and a final experiment stack designed for reproducible research, MLflow tracking, lightweight reporting, and MLOps-ready execution.

The production-facing delivery path is:

```text
experiments/fl-iot-ids-v3
```

---

## Project Overview

Modern IoT networks generate sensitive network traffic that is difficult to centralize safely. This project studies how to train an IDS without moving raw traffic data out of local nodes. Each client trains on its own partition, sends model updates to a Flower server, and receives the next global model after aggregation.

The current validated delivery stack is classical FL (`FedAvg` and `FedProx`) over CICIoT2023. The quantum-inspired part is positioned as the next research layer: feature selection, communication compression, and adaptive resource management for constrained IoT/edge environments.

The final project is built around four engineering goals:

- Privacy-preserving learning: raw CICIoT2023 samples remain local to simulated IoT nodes.
- Non-IID robustness: experiments cover realistic client drift and missing-class scenarios.
- Reproducibility: experiment definitions are frozen in YAML and launched through a registry.
- MLOps traceability: MLflow stores parameters, metrics, artifacts, and generated reports.
- Quantum-inspired evolution: future modules target feature selection, update compression, and resource-aware FL orchestration.

## Repository Positioning

| Area | Role | Path |
| --- | --- | --- |
| Centralized baseline | Single-machine reference model and comparison point | `experiments/baseline-CIC_IOT_2023` |
| FL v1 | Initial Flower-based federated implementation | `experiments/fl-iot-ids-v1` |
| FL v2 | Intermediate and quantum-inspired exploration layer kept for comparison | `experiments/fl-iot-ids-v2` |
| FL v3 | Final delivery stack with registry-driven experiments, reports, and MLflow | `experiments/fl-iot-ids-v3` |

The root README focuses on the final system and keeps earlier versions as historical and scientific context.

---

## System Architecture

The final setup follows a cross-silo FL architecture with one server and three logical clients.

```text
                 +--------------------------------+
                 |          FL Server             |
                 |  Round orchestration           |
                 |  FedAvg / FedProx aggregation  |
                 |  Metrics, artifacts, MLflow    |
                 +---------------+----------------+
                                 |
          -------------------------------------------------
          |                       |                       |
    +-----+------+          +-----+------+          +-----+------+
    |   node1    |          |   node2    |          |   node3    |
    | local fit  |          | local fit  |          | local fit  |
    | local eval |          | local eval |          | local eval |
    +------------+          +------------+          +------------+

Data boundary: raw traffic stays inside each node.
Communication boundary: only model parameters and metrics are exchanged.
Tracking boundary: run metadata and reports are exported to MLflow/output folders.
```

### MLOps and DevOps View

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Data layer | CICIoT2023 preprocessing, scaling, and scenario partitions | `src/data`, `src/scripts/generate_scenarios.py` |
| Experiment layer | Named experiment bundles and reproducible launches | `configs/experiment_registry.yaml`, `src/scripts/run_experiment.py` |
| Training layer | Flower server/client execution, local training, aggregation | `src/fl`, `src/model`, `src/scripts/run_server.py`, `src/scripts/run_client.py` |
| Tracking layer | MLflow runs, resolved configs, summaries, round metrics | `src/tracking`, `src/utils/mlflow_logger.py`, `outputs/mlruns` |
| Reporting layer | CSV, Markdown, and HTML result exports | `src/scripts/build_ablation_table.py`, `outputs/reports` |
| Runtime layer | Local Python execution and Docker Compose orchestration | `requirements.txt`, `environment.yml`, `deployments/docker` |
| Quality layer | Unit/smoke tests and CI workflow | `tests`, `.github/workflows/ci.yml` |

---

## Final Experiment Stack

`fl-iot-ids-v3` is the retained delivery stack. It contains:

- a Flower-based FL server,
- three logical clients: `node1`, `node2`, and `node3`,
- registry-driven experiment definitions,
- scenario-specific non-IID dataset generation,
- PyTorch tabular MLP models,
- class weighting and focal-loss variants,
- MLflow tracking with local file backend,
- lightweight result exports for delivery and reporting.

### Federated Strategies

| Strategy | Status | Purpose |
| --- | --- | --- |
| `FedAvg` | Main baseline | Weighted average of local client updates. |
| `FedProx` | Validated alternative | Adds a proximal term to reduce client drift under severe non-IID data. |
| `SCAFFOLD` | Investigated / unstable | Studied as a drift-correction method, but not retained as the default final pipeline. |

### Data Scenarios

| Scenario | Goal |
| --- | --- |
| `normal_noniid` | Clients keep broad class coverage but with different class proportions. |
| `absent_local` | Some classes are missing from some clients to simulate stronger local blind spots. |
| `rare_expert` | Rare attack knowledge is concentrated on one expert client, exposing bias and specialization issues. |

### Model

The main classifier is a compact MLP for engineered CICIoT2023 tabular features:

```text
Input (28 features)
  -> Linear(28, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Dropout(0.2)
  -> Linear(128, 34)
  -> Output logits
```

An alternate `flat_34_v1style` configuration with `[128, 64]` hidden layers is also available for selected focal-loss experiments.

---

## Quantum-Inspired Roadmap

The repository name and long-term research direction are intentionally quantum-inspired. The validated `fl-iot-ids-v3` stack establishes the reproducible FL/MLOps base first; quantum-inspired modules are then added as controlled research extensions instead of being mixed into the baseline without evidence.

| Objective | Purpose | Target Engineering Outcome |
| --- | --- | --- |
| Quantum Genetic Algorithm (QGA) | Select compact and discriminative CICIoT2023 feature subsets | Reduce input dimensionality and training cost while preserving IDS metrics. |
| Federated Tensor Network (FedTN) | Compress client model updates before aggregation | Lower communication overhead per FL round for bandwidth-limited IoT nodes. |
| Quantum-Inspired Adaptive Resource Management (QIARM) | Rank or schedule clients by resource state and contribution value | Make FL rounds more robust under CPU, latency, and bandwidth constraints. |
| Hybrid FL strategy benchmarking | Compare classical FL against quantum-inspired variants | Keep scientific conclusions measurable and reproducible. |
| Privacy enhancement | Add formal privacy mechanisms such as DP-SGD or update clipping | Move beyond data locality toward quantified privacy guarantees. |

### Next Objectives

The next development objectives are:

1. Complete and freeze the final `fl-iot-ids-v3` benchmark across all registry entries.
2. Extend CI so the final v3 tests and smoke checks run automatically, not only legacy stacks.
3. Add a model/artifact registry convention for scalers, class weights, checkpoints, and reports.
4. Implement QGA as an isolated feature-selection module with before/after ablation results.
5. Prototype FedTN-style update compression and measure bytes per round, latency, and Macro-F1 impact.
6. Design QIARM client scheduling around node health, dataset value, bandwidth, and convergence contribution.
7. Prepare a deployment path from Docker Compose toward edge/Kubernetes orchestration with monitoring.

---

## Pipeline

1. Data preparation
   - Start from the fixed CICIoT2023 export.
   - Fit or reuse the global scaler.
   - Generate label mappings, feature names, and class weights.

2. Scenario generation
   - Build `node1`, `node2`, and `node3` partitions.
   - Apply the selected non-IID strategy.
   - Export raw and processed node data under the experiment-local `data` folder.

3. Federated training
   - Resolve the experiment bundle from `experiment_registry.yaml`.
   - Run Flower simulation with the configured strategy, model, data scenario, and imbalance method.
   - Log run parameters, round metrics, and generated artifacts.

4. Evaluation and reporting
   - Track accuracy, macro metrics, benign recall, false-positive rate, and rare-class recall.
   - Export Markdown, CSV, and HTML reports for final delivery.

---

## Experimental Results

Lightweight exported reports are available under `experiments/fl-iot-ids-v3/outputs/reports/`:

- [Experiment comparison HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_experiment_comparison.html)
- [Full ablation table HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_full_ablation_table.html)
- [10-round summary HTML](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_results_10rounds.html)
- [Ablation table CSV](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_ablation_table.csv)
- [Ablation table Markdown](experiments/fl-iot-ids-v3/outputs/reports/fl_v3_ablation_table.md)

Latest lightweight export currently includes:

| Experiment | Strategy | Scenario | Rounds | Macro-F1 | Accuracy | Rare Recall |
| --- | --- | --- | --- | --- | --- | --- |
| `exp_v3_fedavg_normal_classweights` | `fedavg` | `normal_noniid` | 10 | 0.6521 | 0.7651 | 0.4055 |

### Key Findings

- FedAvg remains the main reference strategy for stable comparison.
- FedProx is valuable when heterogeneity becomes severe, especially in `absent_local`.
- SCAFFOLD was explored but is not considered stable enough for the retained delivery pipeline.
- `rare_expert` shows why sample weighting and rare-class recall must be monitored, not only global accuracy.
- Macro metrics are more informative than accuracy alone because CICIoT2023 classes remain difficult and imbalanced after partitioning.

---

## Reproducibility

The final delivery is configured around fixed experiment assumptions:

- Python version: `3.11`
- random seed: `42`
- final stack: `experiments/fl-iot-ids-v3`
- dataset export expected locally under `data/balancing_v3_fixed300k_outputs`
- main registry: `experiments/fl-iot-ids-v3/configs/experiment_registry.yaml`
- global runtime configuration: `experiments/fl-iot-ids-v3/configs/global.yaml`
- MLflow tracking URI: `experiments/fl-iot-ids-v3/outputs/mlruns`
- heavy datasets, checkpoints, MLflow runs, and serialized models are excluded from Git

For consistent reruns, keep the same dataset export, the same YAML configuration files, and the same Python environment.

---

## How to Run

### Option 1: Conda environment

```powershell
conda env create -f environment.yml
conda activate qfl
```

### Option 2: Local venv for the final stack

```powershell
cd experiments\fl-iot-ids-v3
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run a registry-driven experiment

```powershell
cd experiments\fl-iot-ids-v3
python -m src.scripts.run_experiment --experiment exp_v3_fedavg_normal_classweights
```

Useful registry entries:

- `exp_v3_fedavg_normal_classweights`
- `exp_v3_fedavg_normal_focal`
- `exp_v3_fedavg_normal_focal_weighted`
- `exp_v3_fedprox_absentlocal_classweights`
- `exp_v3_fedavg_rareexpert_focal_weighted`

### Start MLflow UI

```powershell
cd experiments\fl-iot-ids-v3
mlflow ui --backend-store-uri outputs\mlruns --port 5000
```

Open `http://localhost:5000`.

### Generate the ablation table

```powershell
cd experiments\fl-iot-ids-v3
python -m src.scripts.build_ablation_table
```

### Docker Compose execution

```powershell
cd experiments\fl-iot-ids-v3
docker compose -f deployments\docker\docker-compose.yml up --build
```

### Tests

```powershell
cd experiments\fl-iot-ids-v3
python -m pytest tests -v
```

---

## Project Structure

```text
qi-fl-ids-iot/
|-- data/                           # local datasets and fixed exports, not tracked
|-- docs/                           # diagrams and supporting report material
|-- experiments/
|   |-- baseline-CIC_IOT_2023/      # centralized baseline experiments
|   |-- fl-iot-ids-v1/              # initial FL reference version
|   |-- fl-iot-ids-v2/              # intermediate version kept for comparison
|   `-- fl-iot-ids-v3/              # final FL system for delivery
|       |-- configs/                # experiment, model, data, FL, and runtime configs
|       |-- deployments/docker/     # Docker Compose and Dockerfiles
|       |-- outputs/reports/        # lightweight exported reports
|       |-- src/                    # data, model, FL, scripts, tracking, services
|       `-- tests/                  # smoke and unit tests
|-- outputs/                        # shared root-level outputs if needed
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- LICENSE
`-- README.md
```

---

## Documentation

- [FL v3 experiment README](experiments/fl-iot-ids-v3/README.md)
- [Baseline experiment README](experiments/baseline-CIC_IOT_2023/README.md)
- [Global vision report](docs/report/global_vision.pdf)
- [Architecture Mermaid diagram](docs/diaggant.mmd)
- [Changelog](CHANGELOG.md)
- [Contributing guide](CONTRIBUTING.md)

---

## Contributing

Contributions should keep the project reproducible and easy to audit:

- use clear experiment names in `configs/experiment_registry.yaml`,
- keep datasets, checkpoints, MLflow runs, and large artifacts out of Git,
- add or update tests when changing model, data, or FL behavior,
- document new experiments with their scenario, strategy, loss, and expected output,
- prefer small, reviewable commits with explicit intent.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide.

---

## Research & References

### Key Publications

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

2. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys 2020. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)

3. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., and Suresh, A. T. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML 2020. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378)

4. Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., and Ghorbani, A. A. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environments*. Sensors, 23(13), 5941. [DOI:10.3390/s23135941](https://doi.org/10.3390/s23135941)

5. Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Parcollet, T., and Lane, N. D. (2022). *Flower: A Friendly Federated Learning Research Framework*. [arXiv:2007.14390](https://arxiv.org/abs/2007.14390)

6. Han, K. H., and Kim, J. H. (2000). *Genetic Quantum Algorithm and its Application to Combinatorial Optimization Problem*. IEEE Congress on Evolutionary Computation. [DOI:10.1109/CEC.2000.870357](https://doi.org/10.1109/CEC.2000.870357)

### Framework Documentation

- [Flower Documentation](https://flower.ai/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact & Support

- **Author:** Saif Eddinne Boukhatem
- **Role:** Final Year Project (PFE) Student
- **Specialization:** Networks and Artificial Intelligence
- **Institution:** Military Academy
- **Email:** saif.boukhatem2@gmail.com
- **GitHub:** [@SAIFBKKK](https://github.com/SAIFBKKK)

**Advisors:**

- Mrs. Abir GALLAS
- Mr. Med Hechmi Jridi

**Support channels:**

- [GitHub Issues](https://github.com/SAIFBKKK/qi-fl-ids-iot/issues)
- [GitHub Discussions](https://github.com/SAIFBKKK/qi-fl-ids-iot/discussions)
- For sensitive or academic inquiries, contact the author by email.

---

## Acknowledgments

- Military Academy for institutional support and project supervision.
- Canadian Institute for Cybersecurity for the CICIoT2023 dataset.
- Flower team for the federated learning framework.
- PyTorch and MLflow communities for the deep learning and experiment tracking tooling.
- Open-source contributors whose tools support reproducible ML engineering.

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@thesis{boukhatem2026flidsiot,
  author = {Boukhatem, Saif Eddinne},
  title = {Quantum-Inspired Federated Learning for IoT Intrusion Detection},
  school = {Military Academy},
  year = {2026},
  type = {Final Year Project}
}
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

- **Current delivery focus:** `experiments/fl-iot-ids-v3`
- **Next research focus:** QGA, FedTN, QIARM, privacy enhancement, and edge deployment
- **Last updated:** April 2026
