# Quantum-Inspired Federated Learning for IoT Intrusion Detection

[![CI/CD Pipeline](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/actions)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Flower Framework](https://img.shields.io/badge/Flower-1.20%2B-FF6B6B.svg)](https://flower.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version 2.1.0](https://img.shields.io/badge/Version-2.1.0-purple.svg)](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/releases/tag/v2.1.0)
[![Status: Stable](https://img.shields.io/badge/Status-Stable-brightgreen.svg)](#current-status)

**Final Year Project (PFE)** — Computer Science, Networks & Artificial Intelligence  
École Nationale d'Ingénieurs de Tunis (ENIT)

---

## Overview

This project implements a **distributed Intrusion Detection System (IDS)** for IoT environments using **Federated Learning** with quantum-inspired optimization. The system preserves data privacy by keeping raw network data localized on IoT nodes while enabling collaborative model training through secure weight aggregation.

### Core Innovation

Rather than centralizing sensitive network traffic, this framework implements:
- **Federated Learning (FL)** — Models travel to data, not data to servers
- **Quantum-Inspired Algorithms** — QGA, FedTN, QIARM for improved optimization efficiency
- **Non-IID Data Handling** — Realistic heterogeneous IoT network simulation
- **Privacy-Preserving Architecture** — Zero raw data exposure at aggregation points

### Why This Matters

Traditional centralized IDS systems in IoT networks face three critical challenges:
1. **Privacy Risk** — Raw network data centralization violates data sovereignty
2. **Scalability Bottleneck** — Single point of failure; impractical for thousands of edge nodes
3. **Communication Overhead** — High bandwidth cost in bandwidth-constrained IoT networks

This project addresses all three through federated learning combined with quantum-inspired optimization techniques.

---

## Technical Specifications

### System Architecture

```
┌──────────────────────────────────────────────────┐
│         FL Server (FedAvg Aggregation)            │
│              Flower Framework                     │
│                  Port 8080                        │
└─────────────┬──────────────┬──────────────┬───────┘
              │              │              │
         ┌────▼───┐      ┌────▼───┐    ┌────▼───┐
         │ Node 1  │      │ Node 2  │    │ Node 3  │
         │ IoT GW  │      │ IoT GW  │    │ IoT GW  │
         │ FL Client      │ FL Client    │FL Client
         └─────────┘      └─────────┘    └─────────┘

         Privacy: Local data never aggregated
         Communication: Only model weights exchanged
         Computation: Local training per node
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **FL Framework** | Flower | ≥ 1.20 |
| **ML Library** | PyTorch | ≥ 2.0 |
| **Container Orchestration** | Docker Compose | 2.0+ |
| **Dataset** | CIC-IoT-2023 | 34 attack types |
| **Language** | Python | 3.9+ |
| **Experiment Tracking** | MLflow | Latest |
| **Version Control** | Git | Latest |

### Key Metrics

| Metric | Baseline | Target (V2+) |
|--------|----------|-------------|
| Accuracy | 95.12% | 96.5%+ |
| Macro-F1 | 92.01% | 93.5%+ |
| Convergence Rounds | 20-25 | 15-20 |
| Communication Overhead | 5% | <2% (FedTN) |

---

## Development Status

### Current Version: 2.1.0 (Stable)

| Phase | Deliverable | Status | Version |
|-------|-------------|--------|---------|
| V1 | FL Baseline (FedAvg) | Complete | ✓ |
| V2 | Docker Reproducibility | Complete | ✓ |
| V3 | Quantum-Inspired Modules | In Progress | Q2 2026 |
| V4 | Microservices Architecture | Planned | Q3 2026 |
| V5 | Edge Deployment (K8s) | Planned | Q4 2026 |

### Release History

- **2.1.0** (March 2026) — IDS metrics stabilized; Docker validation complete
- **2.0.0** (February 2026) — MLflow integration; multi-client training
- **1.0.0** (January 2026) — Initial FL baseline; local training phase

See [Releases](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/releases) for detailed changelog.

---

## Getting Started

### Prerequisites

```
Operating System: Linux (Ubuntu 20.04+) or macOS (12+)
CPU: 4+ cores
RAM: 8GB minimum (16GB recommended for 3+ clients)
Storage: 20GB for dataset and containers

Software Requirements:
- Python 3.9 or higher
- Docker 20.10 or higher
- Docker Compose 2.0 or higher
- Git
```

### Installation

#### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT.git
cd Quantum-Inspired-Federated-IDS-FOR-IOT/experiments/fl-iot-ids-v1

# Build and launch
docker compose -f deployments/docker/docker-compose.yml up --build

# View logs in real-time
docker compose -f deployments/docker/docker-compose.yml logs -f
```

#### Option 2: Local Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd experiments/fl-iot-ids-v1
pip install -r ../../requirements.txt

# Prepare dataset
python src/scripts/prepare_partitions.py
python src/scripts/preprocess_node_data.py

# Start server (Terminal 1)
python src/run_server.py --config configs/server.yaml

# Start clients (Terminals 2-4)
python src/run_client.py --client-id 1 --config configs/client.yaml
python src/run_client.py --client-id 2 --config configs/client.yaml
python src/run_client.py --client-id 3 --config configs/client.yaml

# View MLflow dashboard (Terminal 5)
mlflow ui --host 0.0.0.0 --port 5000
# Open browser: http://localhost:5000
```

### Verify Installation

```bash
# Check Docker setup
docker --version
docker-compose --version

# Check Python environment
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import flwr; print(f'Flower {flwr.__version__}')"
```

---

## Usage Guide

### Running a Federated Learning Experiment

#### Step 1: Configure Parameters

Edit `experiments/fl-iot-ids-v1/configs/fl_config.yaml`:

```yaml
strategy:
  num_rounds: 25
  num_clients: 3
  fraction_fit: 1.0

training:
  local_epochs: 5
  batch_size: 256
  learning_rate: 0.001
  optimizer: adam
```

#### Step 2: Execute Experiment

```bash
cd experiments/fl-iot-ids-v1

# Docker (recommended)
docker compose -f deployments/docker/docker-compose.yml up --build

# Or locally
python src/run_server.py
# (in parallel terminals)
python src/run_client.py --client-id 1
python src/run_client.py --client-id 2
python src/run_client.py --client-id 3
```

#### Step 3: Monitor Training

```bash
# View experiment metrics
mlflow ui --port 5000
# Navigate to http://localhost:5000 in browser

# View container logs
docker compose logs -f server
docker compose logs -f client-1
```

#### Step 4: Collect Results

```bash
# Results automatically logged to MLflow backend
mlruns/0/*/
├── metrics/          # accuracy, macro_f1, loss per round
├── params/           # hyperparameters
├── artifacts/        # trained model weights
└── tags/             # experiment metadata
```

---

## Dataset

### CIC-IoT-2023

The project uses **CIC-IoT-2023**, a comprehensive IoT attack dataset from the Canadian Institute for Cybersecurity.

**Dataset Characteristics:**
- 34 attack types (DDoS, Brute Force, Spoofing, Mirai, MQTT, CoAP variants, etc.)
- 105 IoT devices across multiple device categories
- 33 engineered network features per sample
- Non-IID data distribution (realistic heterogeneous IoT networks)
- Benign and malicious traffic baseline

**Data Location:**
The dataset is **not** included in this repository due to licensing and size constraints. See [`data/README.md`](data/README.md) for download instructions.

**Expected Directory Layout:**
```
experiments/fl-iot-ids-v1/data/
├── raw/
│   └── CIC-IoT2023.csv
└── processed/
    ├── node1_train.parquet
    ├── node1_test.parquet
    ├── node2_train.parquet
    ├── node2_test.parquet
    ├── node3_train.parquet
    └── node3_test.parquet
```

---

## Project Structure

```
Quantum-Inspired-Federated-IDS-FOR-IOT/
│
├── README.md                                   # This file
├── LICENSE                                     # MIT License
├── requirements.txt                            # Python dependencies
│
├── data/
│   └── README.md                               # Dataset download instructions
│
├── docs/
│   ├── architecture.md                         # System design documentation
│   ├── api.md                                  # API reference
│   ├── deployment.md                           # Deployment guide
│   │
│   ├── architecture/
│   │   ├── diagrams/                           # UML diagrams (.mmd, .png, .svg)
│   │   └── component-diagram.png
│   │
│   └── releases/
│       └── v2.1.0.md                           # Release notes
│
├── experiments/
│   │
│   ├── baseline-CIC_IOT_2023/
│   │   ├── README.md                           # Baseline experiment documentation
│   │   ├── notebooks/
│   │   │   ├── 01_exploratory_data_analysis.ipynb
│   │   │   ├── 02_feature_engineering.ipynb
│   │   │   └── 03_model_training.ipynb
│   │   ├── src/
│   │   │   ├── data_loader.py
│   │   │   ├── models.py
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   ├── results/
│   │   │   ├── baseline_results.json
│   │   │   └── model_artifacts/
│   │   └── figures/
│   │       ├── confusion_matrix.png
│   │       ├── roc_curves.png
│   │       └── feature_importance.png
│   │
│   ├── fl-iot-ids-v1/
│   │   ├── README.md                           # Experiment-specific documentation
│   │   │
│   │   ├── src/
│   │   │   ├── common/
│   │   │   │   ├── config_loader.py
│   │   │   │   ├── logger.py
│   │   │   │   └── metrics.py
│   │   │   │
│   │   │   ├── data/
│   │   │   │   ├── loader.py
│   │   │   │   ├── partitioner.py
│   │   │   │   └── preprocessor.py
│   │   │   │
│   │   │   ├── fl/
│   │   │   │   ├── strategy.py                 # FedAvg strategy
│   │   │   │   ├── server.py
│   │   │   │   └── client.py
│   │   │   │
│   │   │   ├── models/
│   │   │   │   └── ids_model.py                # PyTorch MLP
│   │   │   │
│   │   │   └── scripts/
│   │   │       ├── run_server.py
│   │   │       ├── run_client.py
│   │   │       ├── prepare_partitions.py
│   │   │       └── preprocess_node_data.py
│   │   │
│   │   ├── configs/
│   │   │   ├── fl_config.yaml                  # FL strategy config
│   │   │   ├── server_config.yaml              # Server settings
│   │   │   ├── client_config.yaml              # Client settings
│   │   │   └── dataset_config.yaml             # Dataset paths
│   │   │
│   │   ├── deployments/
│   │   │   ├── docker/
│   │   │   │   ├── base.Dockerfile             # Common base image
│   │   │   │   ├── Dockerfile.server           # Server image
│   │   │   │   ├── Dockerfile.client           # Client image
│   │   │   │   └── docker-compose.yml          # Orchestration
│   │   │   │
│   │   │   └── kubernetes/                     # Planned for V4
│   │   │       └── manifests/
│   │   │
│   │   ├── tests/
│   │   │   ├── test_data_loader.py
│   │   │   ├── test_models.py
│   │   │   ├── test_fl_strategy.py
│   │   │   └── test_integration.py
│   │   │
│   │   ├── notebooks/
│   │   │   ├── 01_data_partitioning.ipynb
│   │   │   └── 02_fl_metrics_analysis.ipynb
│   │   │
│   │   └── docs/
│   │       ├── RUNBOOK.md                      # Step-by-step execution guide
│   │       └── TROUBLESHOOTING.md              # Common issues and solutions
│   │
│   ├── fl-iot-ids-v2/
│   │   ├── README.md                           # Quantum-inspired extensions
│   │   └── src/
│   │       ├── quantum/
│   │       │   ├── qga.py                      # Quantum Genetic Algorithm
│   │       │   ├── fedtn.py                    # Federated Tensor Network
│   │       │   └── qiarm.py                    # Quantum-Inspired Resource Management
│   │       └── ...
│   │
│   └── fl-iot-ids-v3/
│       ├── README.md                           # MLOps and production setup
│       └── src/
│           ├── mlflow_tracker.py
│           ├── ci_cd.yaml
│           └── ...
│
├── shared/
│   └── utilities.py                            # Shared utilities for all versions
│
└── .github/
    └── workflows/
        ├── ci.yml                              # Continuous integration
        ├── docker-build.yml                    # Docker image build
        └── tests.yml                           # Automated testing
```

---

## Configuration

### Server Configuration

`experiments/fl-iot-ids-v1/configs/server_config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 1

strategy:
  num_rounds: 25
  min_fit_clients: 3
  min_evaluate_clients: 3
  min_available_clients: 3
  fraction_fit: 1.0
  fraction_evaluate: 1.0

model:
  input_features: 33
  hidden_layers: [128, 64, 32]
  output_classes: 5
  activation: "relu"

training:
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "cross_entropy"
```

### Client Configuration

`experiments/fl-iot-ids-v1/configs/client_config.yaml`:

```yaml
client:
  client_id: "node1"
  server_address: "localhost:8080"
  
training:
  local_epochs: 5
  batch_size: 256
  learning_rate: 0.001
  
data:
  train_file: "data/processed/node1_train.parquet"
  test_file: "data/processed/node1_test.parquet"
  validation_split: 0.2
```

---

## Performance Results

### Baseline Experiment (Centralized ML)

| Model | Accuracy | Macro-F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Random Forest | 97.8% | 0.912 | 0.940 | 0.892 |
| XGBoost-GPU | 97.1% | 0.901 | 0.934 | 0.878 |
| MLP (PyTorch) | 96.2% | 0.882 | 0.918 | 0.861 |
| Logistic Regression | 89.4% | 0.741 | 0.807 | 0.706 |

### Federated Learning Results (V2.1)

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy (25 rounds) | 95.12% | Converged by round 20 |
| Macro-F1 | 92.01% | Balanced across attack types |
| Convergence Speed | ~20 rounds | vs. 50+ for naive approaches |
| Communication Overhead | 5% | Network bandwidth per round |

Per-attack-type evaluation available in [`experiments/fl-iot-ids-v1/results/`](experiments/fl-iot-ids-v1/results/)

---

## Testing

### Run Unit Tests

```bash
cd experiments/fl-iot-ids-v1

# All tests
pytest tests/ -v

# Specific test module
pytest tests/test_models.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Smoke test: verify all components integrate correctly
python -m pytest tests/test_integration.py -v

# End-to-end: simulate single training round
python tests/test_e2e.py
```

### Validation Checklist

Before deployment, verify:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Docker images build successfully
- [ ] Server and 3 clients can connect
- [ ] Training completes without errors
- [ ] Loss decreases monotonically
- [ ] Metrics logged to MLflow correctly

---

## Documentation

- **[Architecture Guide](docs/architecture.md)** — Detailed system design with UML diagrams
- **[Deployment Guide](docs/deployment.md)** — Step-by-step Docker and Kubernetes setup
- **[API Reference](docs/api.md)** — Complete Python API documentation
- **[Experiment Runbook](experiments/fl-iot-ids-v1/docs/RUNBOOK.md)** — Execution walkthrough
- **[Troubleshooting Guide](experiments/fl-iot-ids-v1/docs/TROUBLESHOOTING.md)** — Common issues and solutions
- **[Release Notes](docs/releases/v2.1.0.md)** — Changelog and migration guide

---

## Contributing

We welcome contributions from the research and engineering community. Please follow these guidelines:

### Contribution Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit changes: `git commit -m "feat: add your feature description"`
4. Push to branch: `git push origin feature/your-feature-name`
5. Open a Pull Request with detailed description

### Code Standards

- Python: PEP 8 (enforce with `flake8` and `black`)
- Type hints required for all functions
- Docstrings for all modules and classes
- Unit tests for new functionality (minimum 80% coverage)
- Commit messages follow conventional commits format

### Areas Needing Contributions

- Quantum-Inspired Optimization Modules (QGA, FedTN, QIARM)
- Kubernetes Deployment Manifests
- Performance Optimization and Benchmarking
- Documentation and Tutorials
- CI/CD Pipeline Enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Troubleshooting

### Common Issues

**Issue: Clients cannot connect to server**
```bash
# Check if server is running
docker logs server

# Verify port binding
netstat -an | grep 8080

# Solution: Restart services
docker compose restart
```

**Issue: Training too slow**
```
Solution: Reduce batch size or epochs in configs/client_config.yaml
- batch_size: 256 → 128
- local_epochs: 5 → 3
```

**Issue: Out of memory errors**
```
Solution: Check system resources
- Minimum 8GB RAM required
- Monitor with: docker stats
- Reduce batch size if running on limited hardware
```

For additional troubleshooting, see [TROUBLESHOOTING.md](experiments/fl-iot-ids-v1/docs/TROUBLESHOOTING.md)

---

## Roadmap

### Completed (V2.1)
- Federated Learning baseline with FedAvg
- Docker Compose deployment
- MLflow experiment tracking
- IDS-specific metrics (F1, precision, recall)

### In Progress (Q2 2026)
- Quantum Genetic Algorithm (QGA) for feature selection
- Federated Tensor Network (FedTN) for gradient compression
- QIARM adaptive resource management
- Enhanced privacy (Differential Privacy integration)

### Planned (Q3-Q4 2026)
- Microservices architecture with message queues
- Kubernetes orchestration
- Edge deployment on real IoT hardware
- Real-time threat response system
- Production MLOps pipeline with monitoring

See [PROJECT.md](docs/PROJECT.md) for detailed timeline.

---

## Research & References

### Key Publications

1. **Federated Learning**
   - McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Aguerri, y. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

2. **Quantum-Inspired Optimization**
   - Han, K. H., & Kim, J. H. (2000). *Genetic quantum algorithm and its application to combinatorial optimization problem*. IEEE CEC. [DOI:10.1109/CEC.2000.870357](https://doi.org/10.1109/CEC.2000.870357)

3. **CIC-IoT-2023 Dataset**
   - Nour, M., Sharafaldin, I., & Ghorbani, A. A. (2023). *A Realistic Cyber Attack Dataset*. Sensors, 23(13), 5941. [DOI:10.3390/s23135941](https://doi.org/10.3390/s23135941)

### Framework Documentation

- [Flower Framework Docs](https://flower.dev/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

### Attribution

This work includes research concepts from:
- Canadian Institute for Cybersecurity (CIC-IoT-2023 dataset)
- Flower Framework development team
- Open-source PyTorch and MLflow communities

---

## Contact & Support

### Author

**Saif Eddinne Boukhatem**  
- Role: Final Year Project (PFE) Student
- Specialization: Networks & Artificial Intelligence
- Institution: **Military Academy**
- Email: [saif.boukhatem2@gmail.com]
- GitHub: [@SAIFBKKK](https://github.com/SAIFBKKK)

### Advisors

- **Academic Advisor:** Mrs.Abir GALLAS
- **Academic Advisor:** Mr. Med Hechmi Jridi

### Support

- **Issues & Bug Reports:** [GitHub Issues](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/issues)
- **Discussions:** [GitHub Discussions](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/discussions)
- **Security Issues:** [SECURITY.md](SECURITY.md) (report privately)

---

## Acknowledgments

- **ENIT** for institutional support and research infrastructure
- **CIC** for the CIC-IoT-2023 dataset
- **Flower Team** for the federated learning framework
- **PyTorch Foundation** for deep learning capabilities
- **Open-source community** for numerous tools and libraries

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@thesis{boukhatem2026qiflids,
  author = {Boukhatem, Saif Eddinne},
  title = {Quantum-Inspired Federated Learning for IoT Intrusion Detection},
  school = {Military Academy},
  year = {2026},
  type = {Final Year Project (Projet de Fin d'Études)}
}
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current Version:** 2.1.0 (Stable)  
**Last Updated:** March 2026

---

**Questions or suggestions?** Open an issue on [GitHub](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/issues) or contact the author.
