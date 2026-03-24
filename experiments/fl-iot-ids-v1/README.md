# Quantum-Inspired Federated Framework for Dynamic IoT Networks

> **Projet de Fin d'Études (PFE)** — Génie Informatique · Réseaux & Intelligence Artificielle  
> Federated Learning-based Intrusion Detection System for IoT/WSN,  
> with Quantum-Inspired extensions (QGA · FedTN · QIARM)

---

## Project Overview

Modern IoT deployments generate massive volumes of network traffic across distributed, resource-constrained devices. Centralized intrusion detection is impractical at this scale — it creates single points of failure, exposes raw traffic data, and cannot adapt to heterogeneous node capabilities.

This project implements a **privacy-preserving, distributed IDS** using Federated Learning, trained on the [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset (34 attack classes, 3 federated nodes). Instead of centralizing raw data, the system uses FL combined with **quantum-inspired optimization techniques** to enable collaborative, privacy-aware model training across edge devices.

This work is developed as a Final Year Project (PFE) in Computer Science — Networks and Artificial Intelligence.

---

## Objectives

- Design a federated learning architecture adapted to IoT constraints
- Integrate quantum-inspired algorithms (QGA, FedTN, QIARM) to improve optimization and convergence
- Preserve data privacy by keeping data localized on IoT nodes
- Reduce communication overhead and energy consumption
- Evaluate performance on realistic heterogeneous IoT network scenarios (CIC-IoT-2023)

---

## Key Concepts

- Internet of Things (IoT) · Wireless Sensor Networks (WSN)
- Federated Learning (FL) · FedAvg aggregation
- Quantum-Inspired Optimization (QGA, QIARM)
- Federated Trust Networks (FedTN)
- Edge Computing · Privacy-Preserving AI
- Distributed Intrusion Detection Systems (IDS)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              FL Server  (FedAvg · Flower)                │
│                      port 8080                           │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
        ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐
        │ Node 1  │    │ Node 2  │    │ Node 3  │
        │ IoT GW  │    │ IoT GW  │    │ IoT GW  │
        │ Flower  │    │ Flower  │    │ Flower  │
        │ Client  │    │ Client  │    │ Client  │
        └─────────┘    └─────────┘    └─────────┘

         Local data stays on each node — only model updates are shared.
```

**Main components:**

- **IoT Nodes (Clients)** — Resource-constrained devices with sensors and local training capabilities
- **Federated Server** — Coordinates training rounds and aggregates local model updates via FedAvg
- **Quantum-Inspired Module** *(v2)* — Enhances aggregation, feature selection, and hyperparameter tuning using QGA · FedTN · QIARM
- **Simulation & Evaluation Layer** — Experiments over different topologies and heterogeneous conditions

Full UML diagrams (component, deployment, sequence, activity) → [`docs/architecture/diagrams/`](docs/architecture/diagrams/)

---

## Experiment Versions

| Version | Description | Dataset | Status |
|---------|-------------|---------|--------|
| [`baseline-CIC_IOT_2023`](experiments/baseline-CIC_IOT_2023/) | Centralized ML baseline — RF, MLP, XGBoost | CIC-IoT-2023 | ✅ Complete |
| [`fl-iot-ids-v1`](experiments/fl-iot-ids-v1/) | Classical FL — FedAvg · 3 nodes · Flower | CIC-IoT-2023 | ✅ Local phase · 🐳 Docker in progress |
| [`fl-iot-ids-v2`](experiments/fl-iot-ids-v2/) | Quantum-Inspired — QGA · FedTN · QIARM | CIC-IoT-2023 | 🔜 Planned |
| [`fl-iot-ids-v3`](experiments/fl-iot-ids-v3/) | MLOps / production — MLflow · CI/CD · Kubernetes | CIC-IoT-2023 | 🔜 Planned |

---

## Key Results — Centralized Baseline

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Random Forest | **97.8%** | **0.91** | Best overall |
| XGBoost | 97.1% | 0.90 | GPU-trained |
| MLP | 96.2% | 0.88 | FL target architecture |
| Logistic Regression | 89.4% | 0.74 | — |

> Federated learning results (per-round accuracy, loss, communication cost) will be reported after Docker validation.

### Evaluation Metrics

- Model accuracy · Macro F1 · Per-class recall
- Convergence speed across FL rounds
- Communication cost per round
- Energy consumption (planned for v2)

---

## Technologies & Tools

| Layer | Technology |
|-------|-----------|
| FL Framework | [Flower](https://flower.ai) ≥ 1.20 |
| Deep Learning | PyTorch ≥ 2.2 (CPU-optimized in containers) |
| Dataset | CIC-IoT-2023 · 34 attack classes · 33 features |
| Containerization | Docker · Docker Compose |
| Configuration | YAML per-node configs |
| Language | Python 3.11 |
| Version Control | Git · GitHub |
| Documentation | Markdown · LaTeX (final report) |

---

## Dataset

The CIC-IoT-2023 dataset is **not included** in this repository (licensing and size constraints).  
See [`data/README.md`](data/README.md) for download instructions and the expected directory layout.

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT.git
cd Quantum-Inspired-Federated-IDS-FOR-IOT/experiments/fl-iot-ids-v1

conda env create -f ../../environment.yml
conda activate fl-iot-ids
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Place CIC-IoT-2023 CSVs in data/raw/ per partition_manifest.json
python src/scripts/prepare_partitions.py
python src/scripts/preprocess_node_data.py
```

### 3. Run locally (without Docker)

```bash
# Terminal 1 — server
python -m src.scripts.run_server --config configs/fl_config.yaml

# Terminals 2 / 3 / 4 — clients
python -m src.scripts.run_client --config configs/fl_config.yaml --node-id node1
python -m src.scripts.run_client --config configs/fl_config.yaml --node-id node2
python -m src.scripts.run_client --config configs/fl_config.yaml --node-id node3
```

### 4. Run with Docker Compose

```bash
cd experiments/fl-iot-ids-v1

# Build the shared image
docker build -f deployments/docker/base.Dockerfile -t fl-iot-ids-v1:latest .

# Launch server + 3 clients
docker compose -f deployments/docker/docker-compose.yml up
```

---

## Repository Structure

```
.
├── data/                          # Local only — never committed (see data/README.md)
├── docs/
│   ├── architecture/diagrams/     # UML diagrams (.mmd · .png · .svg)
│   └── report/                    # Report figures and reference PDF
├── experiments/
│   ├── baseline-CIC_IOT_2023/     # Centralized baseline (notebooks, results, figures)
│   ├── fl-iot-ids-v1/             # ← Active FL experiment
│   │   ├── src/                   # Python source (common, data, fl, model, scripts)
│   │   ├── configs/               # YAML configs (global, FL, model, per-node)
│   │   ├── deployments/docker/    # base · client · server Dockerfiles + Compose
│   │   ├── tests/                 # Unit and smoke tests
│   │   └── docs/                  # Experiment-level documentation and runbooks
│   ├── fl-iot-ids-v2/             # Quantum-Inspired extension (planned)
│   └── fl-iot-ids-v3/             # MLOps / production (planned)
└── shared/                        # Future shared utilities across versions
```

---

## Experimental Methodology

Experiments compare centralized training vs. federated learning, and evaluate classical vs. quantum-inspired optimization approaches across three phases:

1. **Centralized baseline** — establish performance ceiling on full CIC-IoT-2023
2. **Federated v1** — reproduce baseline performance under privacy constraints (FedAvg, 3 nodes)
3. **Quantum-Inspired v2** — improve convergence and efficiency with QGA · FedTN · QIARM

---

## Roadmap

- [x] Centralized ML baseline (Random Forest, MLP, XGBoost-GPU)
- [x] Federated data partitioning — IID, 3 nodes
- [x] Local FL training with Flower (FedAvg)
- [x] Flower API migration (`ServerApp` / `ClientApp`)
- [ ] Docker build validation (`base.Dockerfile`)
- [ ] Docker Compose end-to-end test (server + 3 clients)
- [ ] Federated evaluation metrics (per-round accuracy, loss, comm. cost)
- [ ] QGA feature selection (v2)
- [ ] FedTN trust-weighted aggregation (v2)
- [ ] QIARM adaptive resource management (v2)

---

## Expected Outcomes

- Improved learning efficiency in distributed IoT networks under privacy constraints
- Reduced communication overhead via quantum-inspired aggregation
- Demonstrated potential of quantum-inspired techniques in real-world FL pipelines
- Reproducible, containerized experimental framework for future research

---

## References

- CIC-IoT-2023 Dataset — Canadian Institute for Cybersecurity
- Flower Federated Learning Framework — https://flower.ai
- Nour, M. et al. — *A Realistic Cyber Attack Dataset* (sensors-23-05941)

See [`docs/`](docs/) for the full literature review and architecture documentation.

---

## Author

**Saif Eddinne Boukhatem**  
Final Year Project (PFE) — Computer Science · Networks · Artificial Intelligence  
Encadrant : Mrs. Abir GALLAS . Mr. Med Hechmi Jridi   
Institution : Military Academy

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.