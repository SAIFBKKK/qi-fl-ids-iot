# System Architecture

> **QI-FL-IDS-IoT** — Quantum-Inspired Federated Intrusion Detection System for IoT Networks  
> Engineering vision document — PFE Saif Eddinne Boukhatem

---

## Table of Contents

- [System Architecture](#system-architecture)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [High-Level Architecture](#high-level-architecture)
  - [Node Architecture](#node-architecture)
  - [Server Architecture](#server-architecture)
  - [Data Pipeline](#data-pipeline)
  - [ML Training Pipeline](#ml-training-pipeline)
  - [Feature Alignment Strategy](#feature-alignment-strategy)
  - [FL Communication Protocol](#fl-communication-protocol)
  - [Monitoring \& Observability](#monitoring--observability)
  - [Phase 3 — Quantum-Inspired Layer](#phase-3--quantum-inspired-layer)
  - [Target Architecture — Full Deployment](#target-architecture--full-deployment)
  - [Design Principles](#design-principles)
  - [Target Use Cases](#target-use-cases)

---

## Overview

The system implements a **distributed Intrusion Detection System (IDS)** for heterogeneous IoT networks, based on **Federated Learning (FL)**.

Core design constraint: **raw traffic data never leaves the device**.  
Each IoT node trains a local model on its private data partition. Only model weights are exchanged with the server. The global model is built by aggregation, not by data pooling.

The architecture is built in progressive phases:

| Phase | Scope | Key Component |
|-------|-------|---------------|
| 1 | Centralized baseline | scikit-learn, PyTorch |
| 2 | Federated IDS | Flower (Flwr), FedAvg |
| 3 | Quantum-Inspired optimization | Custom QI layer |
| 4 | Edge deployment | Docker, EdgeX Foundry, MQTT |

---

## High-Level Architecture

```
                    ┌────────────────────────────┐
                    │         FL Server           │
                    │  ┌──────────────────────┐  │
                    │  │  FedAvg Aggregation  │  │
                    │  │  Global Model Update │  │
                    │  │  Round Orchestration │  │
                    │  └──────────────────────┘  │
                    └──────────────┬─────────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
          ┌────────┴──────┐ ┌──────┴───────┐ ┌────┴──────────┐
          │   IoT Node 1  │ │  IoT Node 2  │ │  IoT Node 3   │
          │               │ │              │ │               │
          │  Local Data   │ │  Local Data  │ │  Local Data   │
          │  Local Train  │ │  Local Train │ │  Local Train  │
          │  IDS Model    │ │  IDS Model   │ │  IDS Model    │
          └───────────────┘ └──────────────┘ └───────────────┘
```

**Communication pattern:** pull-based — clients connect to the server, never the reverse.  
**Data flow:** weights only — no raw samples cross node boundaries.

---

## Node Architecture

Each IoT node is a self-contained unit handling its own data lifecycle:

```
┌──────────────────────────────────────────┐
│               IoT Node                   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │         Local Dataset             │   │
│  │  CIC-IoT-2023 partition (non-IID) │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │       Preprocessing Layer        │   │
│  │  • Feature scaling (scaler.pkl)  │   │
│  │  • Label mapping (34 classes)    │   │
│  │  • feature_names.json alignment  │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │     PyTorch DataLoader           │   │
│  │  batch_size=256 · shuffle=True   │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │     Local Training (MLP)         │   │
│  │  optimizer=Adam · epochs=1       │   │
│  │  loss=CrossEntropyLoss           │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │     Flower Client Interface      │   │
│  │  get_parameters() / set_parameters() │
│  │  fit() / evaluate()              │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│           (weights → server)             │
└──────────────────────────────────────────┘
```

---

## Server Architecture

The FL server is stateless between rounds — it holds only the current global model:

```
┌──────────────────────────────────────────┐
│              Federated Server            │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │       Round Orchestration        │   │
│  │  • Client sampling               │   │
│  │  • min_available_clients check   │   │
│  │  • Timeout management            │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │      Model Broadcast             │   │
│  │  • Serialize global weights      │   │
│  │  • Send to selected clients      │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │      FedAvg Aggregation          │   │
│  │  • Collect client weight updates │   │
│  │  • Weighted average by n_samples │   │
│  │  • Produce new global model      │   │
│  └────────────────┬─────────────────┘   │
│                   ↓                      │
│  ┌──────────────────────────────────┐   │
│  │      Global Evaluation           │   │
│  │  • Accuracy · F1-macro           │   │
│  │  • Benign Recall (primary KPI)   │   │
│  │  • MLflow logging                │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

---

## Data Pipeline

```
CIC-IoT-2023 raw CSV (~5.7M samples, 34 classes)
                    │
                    ▼
        ┌───────────────────────┐
        │   Dirichlet partition  │
        │   α = 0.5  (non-IID)  │
        │   3 node splits       │
        └───────────┬───────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   node1/        node2/       node3/
   train.csv    train.csv    train.csv
       │            │            │
       ▼            ▼            ▼
   RobustScaler (shared scaler.pkl)
       │            │            │
       ▼            ▼            ▼
   PyTorch TensorDataset
       │            │            │
       ▼            ▼            ▼
   DataLoader (batch=256)
       │            │            │
       ▼            ▼            ▼
   Local training → weights → FedAvg
```

**Non-IID rationale:** Dirichlet partitioning with α=0.5 simulates realistic IoT deployments where different network segments see different attack distributions. This is the primary source of *local drift* that quantum-inspired optimization will address in Phase 3.

---

## ML Training Pipeline

```
Input features (N × D)
        │
        ▼
RobustScaler normalization
        │
        ▼
┌───────────────────────┐
│   MLP Classifier       │
│  ┌──────────────────┐ │
│  │  Dense (D → 128) │ │
│  │  ReLU            │ │
│  ├──────────────────┤ │
│  │  Dense (128 → 64)│ │
│  │  ReLU            │ │
│  ├──────────────────┤ │
│  │  Dense (64 → C)  │ │  C = 34 (full) / 7 (family) / 2 (binary)
│  │  Softmax         │ │
│  └──────────────────┘ │
└───────────┬───────────┘
            │
            ▼
    CrossEntropyLoss
            │
            ▼
       Adam optimizer
            │
            ▼
    local_epochs = 1
            │
            ▼
    weight update Δw
            │
            ▼ (sent to server)
         FedAvg
```

---

## Feature Alignment Strategy

A critical challenge in federated IDS: each node must produce **identical feature vectors** despite processing data independently.

**Solution:** shared artifacts mounted via Docker volume

```
Docker Volume: /app/artifacts/
  ├── scaler.pkl          ← RobustScaler fitted on global training set
  ├── feature_names.json  ← Ordered feature list (D features)
  ├── label_mapping_34.pkl
  └── class_weights_34.pkl

         ┌─────────────────────────────┐
         │       Shared Volume         │
         └──────┬──────────┬──────┬───┘
                │          │      │
           Node 1       Node 2   Node 3
           (read-only)  (r/o)    (r/o)
```

This guarantees:
- Same feature order across all nodes
- Same scaling transform (no distribution shift from independent fitting)
- Same label space (no label mismatch between nodes)

---

## FL Communication Protocol

```
Round r:
─────────────────────────────────────────────────────────

Server                          Client i
  │                                │
  │──── broadcast(w_global_r) ────►│
  │                                │
  │                         local_train(w_global_r)
  │                                │  epochs=1
  │                                │  loss=CE
  │                                │
  │◄─── send(w_local_i_r, n_i) ───│
  │                                │
  │  aggregate:                    │
  │  w_global_{r+1} =              │
  │    Σ (n_i / N) · w_local_i_r   │
  │                                │
  │──── broadcast(w_global_{r+1}) ►│
  │                                │

─────────────────────────────────────────────────────────
Repeat for num_rounds
```

Where `n_i` = number of samples on node `i`, `N` = total samples across all participating nodes.

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────┐
│              Observability Stack                │
│                                                 │
│  MLflow                                         │
│    • per-round metrics (loss, accuracy, F1)     │
│    • Benign Recall tracking                     │
│    • model artifact versioning                  │
│                                                 │
│  Prometheus                                     │
│    • FL round duration                          │
│    • client availability                        │
│    • aggregation latency                        │
│                                                 │
│  Grafana                                        │
│    • real-time FL convergence dashboard         │
│    • per-class recall heatmap                   │
│    • node participation timeline                │
└─────────────────────────────────────────────────┘
```

---

## Phase 3 — Quantum-Inspired Layer

The quantum-inspired (QI) optimization layer is the **novel scientific contribution** of this PFE. It operates at the aggregation layer without requiring quantum hardware.

**Motivation:** Standard FedAvg is vulnerable to local drift in non-IID settings. QI mechanisms introduce:
- Superposition-based search over aggregation weight space
- Interference-based pruning of low-signal weight components
- Amplitude encoding analogies for feature importance weighting

```
┌──────────────────────────────────────────────┐
│         QI Optimization Layer (Phase 3)       │
│                                              │
│  Input: {w_local_i} from all clients         │
│                                              │
│  ┌────────────────────────────────────┐     │
│  │  Quantum-Inspired Feature Selection │     │
│  │  (amplitude-encoding analogy)      │     │
│  └────────────────┬───────────────────┘     │
│                   ↓                          │
│  ┌────────────────────────────────────┐     │
│  │  QI Aggregation Optimizer          │     │
│  │  (superposition search)            │     │
│  └────────────────┬───────────────────┘     │
│                   ↓                          │
│  ┌────────────────────────────────────┐     │
│  │  Interference-Based Pruning        │     │
│  │  (destructive interference → drop) │     │
│  └────────────────┬───────────────────┘     │
│                   ↓                          │
│  Output: w_global (QI-enhanced)              │
└──────────────────────────────────────────────┘
```

**Primary target:** improve Benign Recall (reduce false positives on normal IoT traffic).  
**Secondary targets:** rare class recovery (Recon, BruteForce, WebAttack).

---

## Target Architecture — Full Deployment

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
│                                                              │
│  ┌────────────────────┐    ┌───────────────────────────┐   │
│  │   EdgeX Foundry    │    │      FL Server             │   │
│  │  ┌──────────────┐  │    │  FedAvg + QI Optimizer    │   │
│  │  │ Core Data    │  │    │  MLflow · Prometheus       │   │
│  │  │ Core Command │  │    └───────────────────────────┘   │
│  │  │ Core Metadata│  │                │                    │
│  │  └──────────────┘  │                │ gRPC (Flower)      │
│  └────────┬───────────┘                │                    │
│           │ MQTT                        │                    │
│    ───────┼────────────────────────────┼───────────────     │
│           │                            │                    │
│  ┌────────┴──────┐  ┌──────────┐  ┌───┴───────────┐       │
│  │ IoT Node 1    │  │IoT Node 2│  │ IoT Node 3    │       │
│  │ Local IDS     │  │Local IDS │  │ Local IDS     │       │
│  │ Flower Client │  │FL Client │  │ FL Client     │       │
│  └───────────────┘  └──────────┘  └───────────────┘       │
│                                                              │
│  ┌───────────────┐   ┌─────────────────────────────────┐   │
│  │   Prometheus  │──►│         Grafana                 │   │
│  │   (metrics)   │   │  FL convergence dashboard       │   │
│  └───────────────┘   └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Privacy preservation** | No raw data leaves any node; weights only |
| **Non-IID robustness** | Dirichlet partitioning + QI aggregation |
| **Modular architecture** | Phase-separated — FL core decoupled from QI layer |
| **Feature consistency** | Shared scaler.pkl + feature_names.json via Docker volume |
| **Reproducibility** | MLflow experiment tracking, seeded partitioning |
| **Edge-readiness** | MLP optimized for TinyML constraints |
| **Observability** | MLflow + Prometheus + Grafana at every layer |

---

## Target Use Cases

- Smart city infrastructure monitoring
- Industrial IoT (IIoT) security
- Smart grid intrusion detection
- Distributed sensor network protection
- Edge computing security fabric

---

*Document maintained alongside active development. Last updated: Phase 2 complete, Phase 3 in progress.*