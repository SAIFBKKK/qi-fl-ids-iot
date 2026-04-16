# FL IoT IDS v2 — Robust Federated Learning Benchmark

Modern experimental framework for **non-IID federated intrusion detection** on IoT data.

## Goals

This version is designed to benchmark:

- **FL algorithms**: FedAvg, FedProx, SCAFFOLD
- **model architectures**: Flat 34-class, Hierarchical
- **data scenarios**: normal_noniid, absent_local, rare_expert
- **imbalance handling**: none, class_weights, focal_loss, light_oversampling
- **tracking and reproducibility**: MLflow + structured configs
- **future extensions**: QIARM, QGA, FedTN

## Design principles

- modern **Flower Framework** foundation using **ServerApp / ClientApp**
- config-driven experimentation
- strict separation of scenarios and outputs
- reproducible runs with seeded configs
- metrics aligned with IDS research:
  - macro-F1
  - benign recall
  - false positive rate
  - critical-family recall
  - convergence stability
  - communication overhead

## Repository layout

- configs/ experiment configuration
- src/ source code
- data/ scenario-separated data
- rtifacts/ shared and scenario artifacts
- outputs/ metrics, logs, reports, figures, checkpoints
- mlruns/ MLflow local tracking
- deployments/ Docker and MLflow deployment files

## Experimental roadmap

### Ablation A — Architecture
- flat_34
- hierarchical

### Ablation B — FL algorithm
- fedavg
- fedprox
- scaffold

### Ablation C — Rare data scenario
- without expert client
- with expert client

### Ablation D — Imbalance handling
- none
- class_weights
- focal_loss
- light_oversampling

## Flower alignment

This project is intentionally aligned with the modern Flower application model:

- ServerApp
- ClientApp
- strategy-based orchestration
- simulation runtime for scalable experiments
- server-side + federated evaluation support

## Current status

Scaffold created. Core implementation starts from configuration and experiment registry.