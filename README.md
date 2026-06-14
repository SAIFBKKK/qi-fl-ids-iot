# QI-FL-IDS-IoT

[![CI](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml)
[![Docs](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docs.yml)
[![Docker Smoke](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docker-smoke.yml/badge.svg)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docker-smoke.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FF6B6B.svg)](https://flower.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20COMING_SOON-20BEFF.svg)](#dataset)
[![Artifacts](https://img.shields.io/badge/Artifacts-External%20COMING_SOON-6f42c1.svg)](#external-artifacts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Framework-orange.svg)](#project-status)

**Quantum-Inspired Federated Framework for Dynamic IoT Networks: Intrusion Detection System for IoT/WSN Security**

QI-FL-IDS-IoT is a research framework for privacy-preserving intrusion detection in IoT and wireless sensor networks. It combines Federated Learning, quantum-inspired feature selection, adaptive aggregation research, compression analysis, Docker microservices, MQTT traffic flow, and monitoring with Prometheus/Grafana.

The final selected practical model is **FedAvg + QGA** using **12 QGA-selected features** on a CICIoT2023-derived processed dataset.

## Project Status

This repository is being prepared as a public research framework. It is not a certified production security product.

- Academic context: final year engineering project, National Engineering Degree in Computer Engineering, Military Academy, Tunisia.
- Author: SLt Saif Eddinne Boukhatem.
- Defense date: 11 June 2026.
- Grade: 16/20.
- Kaggle dataset: `COMING_SOON`.
- External artifacts archive: `COMING_SOON`.

Full datasets and generated artifacts are intentionally not stored in GitHub. Local Phase 3 staging paths exist only for the developer and are not public download URLs.

## Key Features

- Privacy-preserving federated IDS training with Flower.
- Raw IoT traffic remains local; only model updates are exchanged.
- Final FedAvg baseline and FedAvg + QGA feature-selection path.
- QGA feature selection reducing the final L1 binary model to 12 features.
- QIFA adaptive aggregation research path.
- FedTN/MPS structural compression analysis.
- CICIoT2023 preprocessing and federated partitioning.
- Docker Compose deployment with IDS API, MQTT, monitoring, and dashboard services.
- Prometheus/Grafana observability and live-lab demonstration flow.

## Final Selected Model

The final practical model is **FedAvg + QGA**.

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.9480 |
| Attack Recall | 0.9550 |
| FPR | 0.0594 |
| Features | 12 |
| Bandwidth | 7,236,000 B |

The model keeps the federated learning deployment simple while improving feature efficiency and selected metrics over the 28-feature FedAvg baseline.

## Architecture

```text
IoT / WSN clients
  -> local preprocessing
  -> local IDS model training
  -> Flower federated rounds
  -> global model aggregation
  -> final IDS API and live monitoring

Quantum-inspired research modules:
  QGA  -> feature selection
  QIFA -> adaptive aggregation research
  FedTN/MPS -> structural compression analysis
```

The deployment layer uses Docker services for traffic generation/replay, MQTT transport, feature extraction, IDS inference, FL components, monitoring, and dashboarding.

## Repository Structure

```text
qi-fl-ids-iot/
  README.md
  LICENSE
  CITATION.cff
  SECURITY.md
  data/
  docs/
  experiments/
  external_artifacts/
  scripts/
  services/
  shared/
  .github/workflows/
```

Important paths:

- `experiments/qi-fl-ids-iot-final/` - final scientific pipeline and selected model metadata.
- `services/` - Docker/MQTT/API/monitoring microservices.
- `docs/` - public documentation.
- `scripts/` - repository maintenance and helper scripts.
- `data/` - dataset policy and tiny samples only.
- `external_artifacts/` - placeholder only; heavy artifacts live outside GitHub.

## Installation

Use Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r experiments/qi-fl-ids-iot-final/requirements.txt
```

Some legacy experiments have their own requirements. Prefer the final pipeline path first.

## Dataset

Kaggle dataset: `COMING_SOON`

The full dataset is not stored in GitHub. The intended public dataset package is a processed, CICIoT2023-derived package for reproducibility of preprocessing, L1 binary IDS training, federated partitions, and QGA feature analysis.

Users must cite the original CICIoT2023 dataset:

```text
Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A.,
Lu, R., and Ghorbani, A. A. (2023).
CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks
in IoT Environments. Sensors, 23(13), 5941.
DOI: 10.3390/s23135941
```

See [docs/03_dataset.md](docs/03_dataset.md) and [data/README.md](data/README.md).

## Quick Start

Inspect the project:

```powershell
git clone https://github.com/SAIFBKKK/qi-fl-ids-iot.git
cd qi-fl-ids-iot
```

Run lightweight checks only after installing dependencies:

```powershell
python -m pytest tests
```

Run Docker services from `services/` only after creating a local `.env` from `.env.example` and replacing placeholders.

## Training and Evaluation

The final training/evaluation code is under:

```text
experiments/qi-fl-ids-iot-final/
```

Typical workflow:

1. Obtain the processed dataset package when available.
2. Restore dataset files into their expected relative paths.
3. Run preprocessing validation.
4. Run L1 binary IDS training/evaluation.
5. Run federated experiments with Flower.
6. Compare FedAvg, FedAvg + QGA, QIFA, and QIFA + QGA.

Exact generated outputs are externalized and referenced through manifests.

## Federated Learning Pipeline

The FL pipeline uses Flower to simulate distributed IoT clients. Each client trains locally on its partition, sends model updates, and receives aggregated global model parameters.

The final practical selection is FedAvg + QGA because it balances performance, bandwidth, and deployment simplicity.

## Quantum-Inspired Modules

- **QGA**: quantum-inspired genetic algorithm for feature selection. This is part of the final selected model.
- **QIFA**: quantum-inspired adaptive aggregation research path. Useful for analysis, not the final deployed selection.
- **FedTN/MPS**: tensor-network/MPS structural compression analysis. This is not full final FL training.

Limitations:

- QIARM is not part of the final validated implementation.
- Secure aggregation/encrypted model updates are not implemented.
- FedTN/MPS is a structural analysis path, not the final training pipeline.

## Deployment and Live Lab

Deployment assets live mainly under `services/` and the final experiment deployment folders.

Components include:

- MQTT traffic flow.
- IDS API.
- feature extraction.
- Docker Compose services.
- Prometheus/Grafana monitoring.
- live-lab demonstration scripts and documentation.

The live lab validates deployment flow and integration behavior. It does not establish new model accuracy claims.

## Results Summary

| Configuration | Features | Macro-F1 | Attack Recall | FPR | Bandwidth |
| ------------- | -------: | -------: | ------------: | --: | --------: |
| FedAvg | 28 | 0.9407 | 0.9474 | 0.0663 | 8,710,560 B |
| FedAvg + QGA | 12 | 0.9480 | 0.9550 | 0.0594 | 7,236,000 B |
| QIFA | 28 | 0.9454 | 0.9436 | 0.0524 | 8,710,560 B |
| QIFA + QGA | 12 | 0.9471 | 0.9592 | 0.0658 | 7,236,000 B |

The final selected configuration is **FedAvg + QGA**.

## External Artifacts

External artifacts archive: `COMING_SOON`

Generated artifacts were moved outside GitHub during cleanup. They include model checkpoints, scalers, logs, reports, figures, MLflow runs, and deployment bundles.

Local developer archive name:

```text
qi-fl-ids-iot-artifacts-v1-20260612.zip
```

Archive SHA256:

```text
582163c484d070aa7dbf3d8600254465513158bed75f8012d8094aaa8813342d
```

See [docs/artifacts.md](docs/artifacts.md).

## Documentation

Start here:

- [docs/README.md](docs/README.md)
- [docs/01_overview.md](docs/01_overview.md)
- [docs/02_architecture.md](docs/02_architecture.md)
- [docs/03_dataset.md](docs/03_dataset.md)
- [docs/06_federated_learning.md](docs/06_federated_learning.md)
- [docs/07_quantum_inspired_modules.md](docs/07_quantum_inspired_modules.md)
- [docs/08_deployment.md](docs/08_deployment.md)
- [docs/10_results.md](docs/10_results.md)

## Reproducibility

For full reproduction:

1. Clone the repository.
2. Install Python dependencies.
3. Download the processed dataset package when available.
4. Restore external artifacts only when exact reported outputs or deployment bundles are needed.
5. Run documented scripts from the final experiment path.

The repository should remain usable for code review and lightweight tests without full datasets.

## Security and Publication Notes

Do not commit:

- `.env` files.
- MQTT password files.
- Kaggle credentials.
- private keys.
- packet captures.
- private lab inventories.

This project does not implement encrypted model updates or production-grade secure aggregation. See [SECURITY.md](SECURITY.md) and [docs/security_publication_checklist.md](docs/security_publication_checklist.md).

## Citation

If you use this project, cite it with [CITATION.cff](CITATION.cff) and cite CICIoT2023.

## License

The root repository is released under the [MIT License](LICENSE).

Some nested legacy experiment metadata still declares Apache-2.0 and should be aligned before a formal public release. See [docs/release_notes_license.md](docs/release_notes_license.md).

## Author and Acknowledgements

Author: ** Saif Eddinne Boukhatem**

Academic context: final year engineering project, National Engineering Degree in Computer Engineering .

Acknowledgements go to the academic supervisors, reviewers, open-source communities behind Flower, PyTorch, Docker, Prometheus, Grafana, and the CICIoT2023 dataset authors.
