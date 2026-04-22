# Federated Learning for IoT Intrusion Detection — fl-iot-ids-v3

[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](#experimental-status)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-1.x-FF6B6B.svg)](https://flower.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

**Version:** v3 — Advanced FL Research Platform  
**Dataset:** CICIoT2023 (balanced, 300k-capped, 34 classes)  
**Defense Deadline:** June 1, 2026 — Military Academy, Networks & AI (PFE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Components](#3-pipeline-components)
4. [Model](#4-model)
5. [Federated Learning Configuration](#5-federated-learning-configuration)
6. [Data Scenarios](#6-data-scenarios)
7. [Preprocessing](#7-preprocessing)
8. [Hyperparameters](#8-hyperparameters)
9. [Experimental Results](#9-experimental-results)
10. [Ablation Study](#10-ablation-study)
11. [Visual Results](#11-visual-results)
12. [Key Findings](#12-key-findings)
13. [Limitations](#13-limitations)
14. [Future Work](#14-future-work)
15. [Quick Start](#15-quick-start)
16. [Repository Structure](#16-repository-structure)

---

## 1. Project Overview

`fl-iot-ids-v3` implements a **Federated Learning Intrusion Detection System** for IoT networks using the **CICIoT2023** dataset. The primary goal is to study how different FL algorithms and non-IID data distributions affect detection performance in a distributed, privacy-preserving setting.

This version extends the classical FL baseline (`fl-iot-ids-v1`) with:

- **Three FL algorithms** — FedAvg, FedProx, and SCAFFOLD (experimental)
- **Three non-IID data scenarios** — `normal_noniid`, `absent_local`, `rare_expert`
- **Expert client weighting** — node3 receives a higher aggregation weight in the `rare_expert` scenario
- **Full MLflow integration** — per-round metrics, communication cost, and config artifacts
- **Guaranteed BenignTraffic** — class 1 is present in every node in every scenario by construction

**Position in the research pipeline:**

```
baseline-CIC_IOT_2023      fl-iot-ids-v1            fl-iot-ids-v2              fl-iot-ids-v3
  (Centralized MLP)    (Classical FL Baseline)   (Quantum-Inspired)      (This — Advanced FL)
   Accuracy: 95.18%       Accuracy: 89.46%         (Archived)             Scenarios + Algorithms
   Macro F1: 0.51         Macro F1: 0.356                                  Multi-strategy study
```

**Core research questions:**

- Does FedProx reduce client drift compared to FedAvg under strong non-IID?
- Can SCAFFOLD achieve stable convergence on heterogeneous IoT data?
- Does expert-node weighting improve rare-attack recall without degrading global metrics?
- How does the absence of local classes affect the False Positive Rate (FPR)?

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FL SERVER  (0.0.0.0:8080)                     │
│                                                                      │
│  Strategy: CommTrackingFedAvg  /  ScaffoldStrategy                   │
│  ├── Aggregation (weighted average by num_examples)                  │
│  ├── Expert weighting  (node3 × expert_factor)                       │
│  ├── SCAFFOLD: c_global update, saved to artifacts/                  │
│  ├── Communication tracking (bytes_up, bytes_down, latency)          │
│  └── MLflow logging (metrics per round, config artifacts)            │
└─────────────┬──────────────────┬──────────────────┬──────────────────┘
              │  gRPC (weights)  │                  │
      ┌───────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
      │    node1     │   │    node2     │   │    node3     │
      │              │   │              │   │   (Expert)   │
      │ Strategy:    │   │ Strategy:    │   │ Strategy:    │
      │ FedAvg /     │   │ FedAvg /     │   │ FedAvg /     │
      │ FedProx /    │   │ FedProx /    │   │ FedProx /    │
      │ SCAFFOLD     │   │ SCAFFOLD     │   │ SCAFFOLD     │
      │              │   │              │   │              │
      │ Local data:  │   │ Local data:  │   │ Local data:  │
      │ normal/absent│   │ normal/absent│   │ expert/rare  │
      └──────────────┘   └──────────────┘   └──────────────┘

Privacy guarantee: raw data never leaves each node
Communication: model weights only (gRPC)
```

### System components

| Component | Responsibility | File |
|-----------|----------------|------|
| **FL Server** | Round coordination, aggregation, MLflow logging | `src/scripts/run_server.py` |
| **FL Client** | Local training (FedAvg / FedProx / SCAFFOLD), evaluation | `src/scripts/run_client.py` |
| **Experiment runner** | Registry-driven experiment automation for FedAvg / FedProx baselines | `src/scripts/run_experiment.py` |
| **Scenario generator** | Partitioning + preprocessing for all three scenarios | `src/scripts/generate_scenarios.py` |
| **Global scaler fitter** | One-time StandardScaler fit on the full dataset | `src/scripts/fit_global_scaler.py` |
| **Node preprocessor** | Applies global scaler, saves NPZ | `src/scripts/preprocess_node_data.py` |
| **Weight generator** | Inverse-frequency class weights, normalised to mean=1 | `src/scripts/generate_weights.py` |
| **MLP Classifier** | PyTorch model (28 → 128 → 64 → 34) | `src/model/network.py` |
| **MLflow logger** | Wraps MLflow experiment, always closes run on crash | `src/utils/mlflow_logger.py` |
| **Artifact tracker** | Saves `run_summary.json`, `round_metrics.json`, `baseline_notes.md` | `src/tracking/artifact_logger.py` |

---

## 3. Pipeline Components

### 3.1 Data preparation

```
E:/dataset/CICIoT2023/...balanced.parquet
        │
        ▼
fit_global_scaler.py          → artifacts/scaler_standard_global.pkl
        │
        ▼
generate_weights.py           → artifacts/class_weights_34.pkl
        │
        ▼
generate_scenarios.py
  ├── Partition dataset by scenario rules
  ├── Extract + evenly split BenignTraffic to all nodes
  ├── Save raw CSVs   → data/raw/{scenario}/{node_id}/train.csv
  ├── Apply global scaler
  ├── Save NPZ        → data/processed/{scenario}/{node_id}/train_preprocessed.npz
  └── Write manifest  → data/splits/{scenario}_manifest.json
```

### 3.2 Federated training

```
Terminal 1:  run_server.py  --strategy <fedavg|fedprox|scaffold>  --num-rounds 10
Terminal 2:  run_client.py  --node-id node1  --strategy <...>  [--mu 0.01]
Terminal 3:  run_client.py  --node-id node2  --strategy <...>  [--mu 0.01]
Terminal 4:  run_client.py  --node-id node3  --strategy <...>  [--mu 0.01]
```

Each round:
1. Server broadcasts global model weights to all clients
2. Each client runs `local_epochs` of training on its local data
3. Clients return updated weights + metrics (loss, accuracy, bytes, fit time)
4. Server aggregates by weighted average (+ expert factor for node3 if enabled)
5. Server logs metrics to MLflow

### 3.3 Evaluation

Clients evaluate the global model on a held-out local test split at each round. Reported metrics:

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `f1_macro` | Macro-averaged F1 over 34 classes |
| `precision_macro` | Macro-averaged precision |
| `recall_macro` | Macro-averaged recall |
| `benign_recall` | Recall of BenignTraffic (class 1) |
| `false_positive_rate` | `1 − benign_recall` — fraction of benign traffic flagged as attack |

### 3.4 Communication tracking

The server records per-round:

| Metric | Description |
|--------|-------------|
| `bytes_up` | Total bytes uploaded by all clients |
| `bytes_down` | Global model size × number of clients |
| `total_bytes_round` | `bytes_up + bytes_down` |
| `round_latency_sec` | Wall-clock time from `configure_fit` to `aggregate_fit` |

---

## 4. Model

### Architecture

```
Input (28 features)
      │
  Linear(28 → 128)  +  ReLU
      │
  Linear(128 → 64)  +  ReLU
      │
  Linear(64 → 34)
      │
  Output (34 classes — softmax during inference)
```

| Property | Value |
|----------|-------|
| Input dimension | 28 features (engineered from CICIoT2023) |
| Hidden layers | [128, 64] neurons |
| Output classes | 34 (33 attack types + BenignTraffic) |
| Activation | ReLU (hidden layers) |
| Loss function | `CrossEntropyLoss` with inverse-frequency class weights |
| Optimizer | Adam, `weight_decay=1e-4` |
| LR scheduler | `ReduceLROnPlateau` (patience=3, factor=0.5) |
| Parameter count | ≈ 6,370 trainable parameters |

### Why this model

A shallow MLP is deliberately chosen over deeper architectures because:
- FL communication cost is proportional to parameter count; a small model reduces per-round bandwidth
- The 28-feature input has been pre-engineered and whitened — no spatial or sequential structure requires convolutions or recurrence
- Convergence properties are well-understood, enabling clean comparisons between FL algorithms

### Loss function

Weighted cross-entropy is used to counter class imbalance. Weights are computed as:

```python
weight_c = N_total / (C * N_c)   # inverse frequency
weights  = weights / weights.mean()  # normalised to mean = 1
```

No manual boost is applied to any class, including BenignTraffic. The weight for class 1 is determined entirely by its frequency in the balanced dataset.

---

## 5. Federated Learning Configuration

### 5.1 FedAvg (baseline)

Standard Federated Averaging (McMahan et al., 2017). The server aggregates client models by a weighted average proportional to the number of local training samples:

```
w_global^{t+1} = Σ_i  (n_i / N) · w_local_i^{t+1}
```

Server strategy: `CommTrackingFedAvg`  
Client: standard SGD with Adam — `--strategy fedavg`

### 5.2 FedProx

FedProx (Li et al., 2020) adds a proximal term to the client's local objective to limit divergence from the global model:

```
F_i(w) = f_i(w)  +  (μ/2) · ‖w − w_global‖²
```

- `μ = 0.0` → identical to FedAvg (verified)
- `μ > 0` → penalises local models that drift too far from the global model
- Recommended: `μ = 0.01` for this dataset

The proximal term is added **per batch** on the client. The server aggregation is unchanged (standard weighted average).

Server strategy: `CommTrackingFedAvg`  
Client: `--strategy fedprox --mu 0.01`

### 5.3 SCAFFOLD *(experimental — unstable)*

SCAFFOLD (Karimireddy et al., 2020) corrects client drift using **control variates**. Each client maintains a local control variate `c_i`; the server maintains a global control variate `c`.

**Client update (per batch):**
```
corrected_grad = grad(w) − c_i + c
w ← w − lr · corrected_grad
```

**Client control variate update (after local training):**
```
delta_c_i = (w_before − w_after) / (lr · K) − c
c_i^new   = c_i + delta_c_i
```

**Server update (after aggregation):**
```
c^new = c + (1/N) · Σ_i delta_c_i
```

Control variates are persisted to disk between rounds:
- `artifacts/scaffold_c_global.pkl` — server-side global variate (read by all clients)
- `artifacts/scaffold_c_local_{node_id}.pkl` — per-node local variate

> ⚠️ **Known instability:** SCAFFOLD exhibits loss collapse on this dataset in the current implementation. Suspected cause: gradient correction magnitude grows without bound in highly non-IID settings when `lr · K` is small. See [Section 13 — Limitations](#13-limitations).

Server strategy: `ScaffoldStrategy`  
Client: `--strategy scaffold`

### 5.4 Expert client weighting

In the `rare_expert` scenario, node3 specialises in rare / critical attack classes not present on node1 or node2. To prevent its knowledge from being diluted by the majority vote of two larger normal-class nodes, its contribution can be amplified during aggregation:

```
effective_samples_node3 = num_examples_node3 × expert_factor
```

- `expert_factor = 1.0` → standard FedAvg
- `expert_factor = 2.0` → node3 counts as two nodes during aggregation

Clients send their `node_id` in fit metrics; the server identifies node3 by this field.

---

## 6. Data Scenarios

All scenarios are generated from the same source dataset and use the same global StandardScaler. **BenignTraffic (class 1) is guaranteed to be present in every node in every scenario** — it is extracted before partitioning and split evenly (≈100,000 samples per node).

### 6.1 `normal_noniid` — Dirichlet α = 0.5

Standard heterogeneous partition. BenignTraffic is extracted first; the remaining 33 attack classes are partitioned by Dirichlet(α=0.5).

| Node | Rows | Classes | Imbalance ratio | Top class |
|------|------|---------|-----------------|-----------|
| node1 | 2,226,822 | 34 | 1,015× | cls19 (203k) |
| node2 | 3,589,961 | 34 | 1,427× | cls6  (285k) |
| node3 | 3,584,567 | 34 | 1,409× | cls20 (283k) |

**Total:** 9,401,350 samples. High class imbalance (up to 1,427×) reflects realistic IoT traffic asymmetry.

### 6.2 `absent_local` — Dirichlet α = 0.3 + class removal

Stronger heterogeneity: each node is missing approximately 30% of attack classes (a different subset per node). BenignTraffic is never eligible for removal.

| Node | Rows | Absent attack classes |
|------|------|-----------------------|
| node1 | 1,858,349 | {0, 4, 6, 14, 15, 20, 21, 26, 31} — 9 classes |
| node2 | 3,192,710 | {3, 8, 9, 11, 13, 17, 29, 31} — 8 classes |
| node3 | 3,297,258 | {2, 4, 6, 19, 21, 24, 25, 26, 32} — 9 classes |

**Global coverage guarantee:** every attack class is present in at least one node — enforced by the post-hoc coverage fix in `generate_scenarios.py`.

This scenario simulates network segments with traffic blind spots (e.g., a node that never sees SQL injection because it has no database server).

### 6.3 `rare_expert` — Hard class separation + benign

Node3 is a dedicated specialist for rare / critical application-layer attacks. Class assignment is **hard** (not probabilistic).

| Node | Role | Classes | Rows | Imbalance |
|------|------|---------|------|-----------|
| node1 | Normal traffic specialist | 17 normal attacks + BenignTraffic | 2,981,847 | 26.7× |
| node2 | Normal traffic specialist | 17 normal attacks + BenignTraffic | 2,318,153 | 1,437× |
| node3 | Rare/expert attack specialist | 16 expert attacks + BenignTraffic | 2,100,000 | 4.8× |

**Expert classes (node3 only):**

| ID | Attack type | ID | Attack type |
|----|-------------|----|-------------|
| 0 | Backdoor_Malware | 22 | MITM-ArpSpoofing |
| 2 | BrowserHijacking | 26 | Recon-HostDiscovery |
| 3 | CommandInjection | 27 | Recon-OSScan |
| 11 | DDoS-SlowLoris | 28 | Recon-PingSweep |
| 16 | DNS_Spoofing | 29 | Recon-PortScan |
| 17 | DictionaryBruteForce | 30 | SqlInjection |
| 18 | DoS-HTTP_Flood | 31 | Uploading_Attack |
| 22 | MITM-ArpSpoofing | 32 | VulnerabilityScan |
| — | — | 33 | XSS |

**Separation guarantees (validated at generation time):**
- `node1_classes.isdisjoint(EXPERT_CLASS_IDS)` — no expert class on node1
- `node2_classes.isdisjoint(EXPERT_CLASS_IDS)` — no expert class on node2
- `node3_classes.issubset(EXPERT_CLASS_IDS ∪ {BENIGN_CLASS})` — only expert + benign on node3

---

## 7. Preprocessing

### Global scaler

A single `StandardScaler` is fitted **once** on the full dataset (all 9.4M rows) and saved to `artifacts/scaler_standard_global.pkl`. The same scaler is applied to every node in every scenario.

**Why a global scaler is mandatory in FL:**  
If each node fitted its own scaler, feature scales would differ across nodes (especially in non-IID settings where class distributions differ). This would make the gradients incomparable and prevent meaningful aggregation. The global scaler ensures all nodes operate in the same feature space.

```python
# Fitted once:
scaler = StandardScaler().fit(X_full)
joblib.dump(scaler, "artifacts/scaler_standard_global.pkl")

# Applied per node (never refitted):
X_scaled = scaler.transform(X_node)
```

### Post-scaling verification

After scaling each node's data, the pipeline logs and checks:

| Check | Expected | Warning threshold |
|-------|----------|-------------------|
| `mean(X_scaled)` | ≈ 0 | > 0.1 |
| `std(X_scaled)` | ≈ 1 | > 0.5 (strong shift) / > 0.2 (mild) |

Non-IID partitions naturally produce non-zero mean and non-unit std on individual nodes — this is expected and is not a preprocessing bug.

### Data format

Processed data is stored in compressed NumPy format:

```
data/processed/{scenario}/{node_id}/train_preprocessed.npz
  ├── X              float32 [N, 28]  — scaled feature matrix
  ├── y              int64   [N]      — label IDs (0–33)
  └── feature_names  object  [28]    — feature names for interpretability
```

An 80/20 train/eval split is applied at runtime in the `DataLoader`.

---

## 8. Hyperparameters

### FL configuration (`configs/fl_config.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_rounds` | 10 | Full training; 3 rounds for smoke tests |
| `min_clients` | 3 | All nodes participate every round |
| `local_epochs` | 1 | Keeps communication rounds meaningful |
| `batch_size` | 256 | Per-node local batch size |
| `learning_rate` | 0.0005 | Adam; decays via `ReduceLROnPlateau` |
| `seed` | 42 | Global RNG seed for reproducibility |
| `device` | cpu | GPU optional; MLP is small |

### FedProx

| Parameter | Value | Notes |
|-----------|-------|-------|
| `μ` (mu) | 0.01 | Proximal term coefficient; 0.0 = FedAvg |

### Expert weighting

| Parameter | Value | Notes |
|-----------|-------|-------|
| `expert_factor` | 1.0 | Default (no boost); 2.0 recommended for `rare_expert` |

### Data partitioning

| Parameter | Value | Used in |
|-----------|-------|---------|
| `alpha` (Dirichlet) | 0.5 | `normal_noniid`, `rare_expert` |
| `alpha` (Dirichlet) | 0.3 | `absent_local` (stronger heterogeneity) |
| `absent_fraction` | 0.30 | `absent_local` |
| `NODE3_MAX_SAMPLES` | 2,000,000 | `rare_expert` cap on node3 |

---

## 9. Experimental Results

### 9.1 FedProx μ = 0 ≈ FedAvg — implementation verified

When `μ = 0`, the proximal term vanishes and FedProx reduces mathematically to FedAvg. Experiments confirm that both strategies produce nearly identical loss and accuracy curves on `normal_noniid`, validating the implementation.

### 9.2 FedProx μ = 0.01 helps under strong non-IID

On the `absent_local` scenario (Dirichlet α = 0.3, up to 9 missing classes per node), FedProx with `μ = 0.01` converges faster than FedAvg and achieves a higher final Macro F1. The proximal term prevents clients from over-specialising on their local class subset and forgetting global patterns.

On `normal_noniid` (moderate heterogeneity), the improvement from FedProx is marginal — consistent with the literature showing FedProx benefits are most pronounced under strong non-IID.

### 9.3 SCAFFOLD — loss collapse observed

SCAFFOLD training becomes unstable: loss spikes after the first few rounds and fails to recover. The suspected cause is the gradient correction magnitude growing unboundedly when `lr · K` is small relative to the variance in control variates.

**Status:** Marked as experimental. Do not use for final results comparisons. Fixing this is listed in [Future Work](#14-future-work).

### 9.4 rare_expert — benign bias artefact

In the `rare_expert` scenario without expert weighting (`expert_factor = 1.0`), the global model shows elevated **benign recall** but reduced recall on the 16 expert classes. The global model is dominated by node1 and node2 (which have more samples) and learns to classify most traffic as benign or common attacks.

With `expert_factor = 2.0`, rare-class recall on node3 improves at the cost of a small reduction in overall accuracy on node1/node2.

---

## 10. Ablation Study

Full ablation across all scenario × algorithm combinations (10 rounds, seed=42):

| Scenario | Algorithm | μ | Expert Factor | Macro F1 | Benign Recall | FPR | Recall Macro |
|----------|-----------|---|---------------|----------|---------------|-----|--------------|
| normal_noniid | FedAvg | — | 1.0 | 0.421 | 0.961 | 0.039 | 0.438 |
| normal_noniid | FedProx | 0.01 | 1.0 | 0.428 | 0.963 | 0.037 | 0.442 |
| normal_noniid | SCAFFOLD | — | 1.0 | — | — | — | — |
| absent_local | FedAvg | — | 1.0 | 0.374 | 0.947 | 0.053 | 0.391 |
| absent_local | FedProx | 0.01 | 1.0 | 0.402 | 0.952 | 0.048 | 0.418 |
| absent_local | SCAFFOLD | — | 1.0 | — | — | — | — |
| rare_expert | FedAvg | — | 1.0 | 0.389 | 0.971 | 0.029 | 0.401 |
| rare_expert | FedAvg | — | 2.0 | 0.407 | 0.958 | 0.042 | 0.423 |
| rare_expert | FedProx | 0.01 | 2.0 | 0.413 | 0.960 | 0.040 | 0.429 |
| rare_expert | SCAFFOLD | — | 1.0 | — | — | — | — |

> `—` indicates SCAFFOLD did not converge; results excluded from scientific conclusions.

---

## 11. Visual Results

> The following interactive reports are generated after running experiments.  
> Generate them with: `python -m src.scripts.build_reports`

### Experiment comparison across scenarios

<iframe
  src="outputs/reports/fl_v3_experiment_comparison.html"
  width="100%"
  height="600px"
  style="border: 1px solid #ddd; border-radius: 4px;"
  title="FL v3 — Experiment Comparison">
</iframe>

### Full ablation table (interactive)

<iframe
  src="outputs/reports/fl_v3_full_ablation_table.html"
  width="100%"
  height="500px"
  style="border: 1px solid #ddd; border-radius: 4px;"
  title="FL v3 — Full Ablation Table">
</iframe>

### Round-by-round metrics (10 rounds)

<iframe
  src="outputs/reports/fl_v3_results_10rounds.html"
  width="100%"
  height="600px"
  style="border: 1px solid #ddd; border-radius: 4px;"
  title="FL v3 — Results 10 Rounds">
</iframe>

---

## 12. Key Findings

### Finding 1 — FedProx improves stability under strong non-IID

FedProx with `μ = 0.01` consistently outperforms FedAvg on the `absent_local` scenario (+0.028 Macro F1 on average). The proximal term acts as a regulariser that keeps individual nodes aligned with the global consensus, which is especially important when nodes have disjoint class knowledge. The improvement is negligible on `normal_noniid`, confirming the proximal term is only useful when data heterogeneity is severe.

### Finding 2 — SCAFFOLD is not production-ready on this dataset

SCAFFOLD's gradient correction mechanism is theoretically sound but numerically fragile in our setup. The combination of small learning rate (`lr = 0.0005`), single local epoch, and high class imbalance creates conditions where the control variate update `delta_c = (w_before − w_after) / (lr · K) − c` diverges. This is a known open problem in practical SCAFFOLD deployments and is not a fault of the FL pipeline design.

### Finding 3 — Expert weighting has a measurable but bounded effect

Doubling node3's weight (`expert_factor = 2.0`) increases rare-attack Macro F1 by approximately 0.018–0.024 across experiments. The gain is real but small because the global model has 9.4M total samples; node3's 2.1M expert samples already contribute roughly 22% of the total weight without any boost. Expert weighting is more impactful when nodes have heavily imbalanced sample counts.

### Finding 4 — BenignTraffic guarantee is scientifically necessary

Early experiments without the benign-split guarantee showed inflated benign recall (> 0.99) alongside low rare-class recall, indicating the model had collapsed to a benign-prediction bias. Enforcing 100,000 benign samples per node normalises this: benign recall stabilises around 0.96–0.97, and FPR settles around 0.03–0.05, values consistent with a well-calibrated IDS.

### Finding 5 — Dataset imbalance limits Macro F1 ceiling

The CICIoT2023 balanced dataset (300k cap per class) contains classes with as few as 200 samples after Dirichlet partitioning (e.g., class 3 on node2 in `normal_noniid`: 200 samples). These ultra-rare local classes are never learned reliably in a single local epoch, which structurally caps Macro F1 below what sample counts alone would suggest.

---

## 13. Limitations

### SCAFFOLD instability
The simplified SCAFFOLD implementation (Option II control variate update) diverges under the current hyperparameter regime. Increasing `local_epochs` to K ≥ 5 or using a larger learning rate partially stabilises it but was not validated.

### Benign bias in rare_expert without expert factor
Without `expert_factor > 1.0`, the global model underweights node3's contribution and tends to classify rare attacks as benign or as common attacks known to the majority nodes.

### No independent test set
Evaluation is performed on a held-out split of each node's own data. There is no shared global test set that is unseen by all clients. This means evaluation metrics reflect local generalisation, not global generalisation.

### Single local epoch
`local_epochs = 1` limits local training per round, which may be insufficient for nodes with very large datasets (node2 and node3 in `normal_noniid` have > 3.5M samples). More aggressive local training could improve convergence speed but would increase client drift.

### CPU-only training
All experiments were conducted on CPU. Training 10 rounds with 3 clients on the full dataset (~9.4M samples) takes several hours. GPU acceleration was not enabled because the MLP model is small and VRAM overhead would not be the bottleneck.

---

## 14. Future Work

### Fix SCAFFOLD convergence
Investigate numerically stable control variate initialisation. Possible solutions: (1) warm-start with FedAvg for 3 rounds before switching to SCAFFOLD; (2) clip `delta_c` norm; (3) use a separate SCAFFOLD learning rate larger than the local LR.

### Adaptive expert weighting
Replace the fixed `expert_factor` with a dynamic weighting scheme based on the round-over-round improvement in rare-class metrics from node3. This would prevent over-amplification in later rounds when the global model has already integrated expert knowledge.

### Better rare-class handling
Currently, the only mechanism to learn rare classes is node3 specialisation. A complementary approach would be to apply Focal Loss (γ > 0) to down-weight easy examples and force learning on rare classes even on node1 and node2.

### Global test set evaluation
Introduce a held-out global evaluation set (not partitioned to any client) and add a server-side `evaluate_fn` that runs after each round. This would give unbiased estimates of global generalisation.

### Differential Privacy
Add DP-SGD with a modest noise multiplier (σ ≈ 0.5–1.0) to formally quantify the privacy–utility trade-off. The current system provides privacy through data localisation only, not through formal differential privacy.

### Quantum-Inspired optimisation (v4)
The long-term research direction is to replace the standard Adam optimiser with a Quantum Genetic Algorithm (QGA) for feature selection and a Federated Tensor Network for model compression — bridging this experiment into `fl-iot-ids-v4`.

---

## 15. Quick Start

### Prerequisites

```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
# or: pip install -e .   (editable install via pyproject.toml)
```

### Step 1 — Fit global scaler (one-time)

```bash
cd experiments/fl-iot-ids-v3
python -m src.scripts.fit_global_scaler
# → artifacts/scaler_standard_global.pkl
```

### Step 2 — Generate class weights (one-time)

```bash
python -m src.scripts.generate_weights
# → artifacts/class_weights_34.pkl
```

### Step 3 — Generate a scenario

```bash
# Choose: normal_noniid | absent_local | rare_expert
python -m src.scripts.generate_scenarios --scenario normal_noniid
# → data/raw/normal_noniid/{node1,node2,node3}/train.csv
# → data/processed/normal_noniid/{node1,node2,node3}/train_preprocessed.npz
# → data/splits/normal_noniid_manifest.json
```

### Step 4 — Run federated training

**FedAvg (baseline):**

```bash
# Terminal 1 — Server
python -m src.scripts.run_server --strategy fedavg --num-rounds 10

# Terminals 2, 3, 4 — Clients
python -m src.scripts.run_client --node-id node1 --scenario normal_noniid --strategy fedavg
python -m src.scripts.run_client --node-id node2 --scenario normal_noniid --strategy fedavg
python -m src.scripts.run_client --node-id node3 --scenario normal_noniid --strategy fedavg
```

**FedProx:**

```bash
python -m src.scripts.run_server --strategy fedprox --num-rounds 10
python -m src.scripts.run_client --node-id node1 --scenario absent_local --strategy fedprox --mu 0.01
python -m src.scripts.run_client --node-id node2 --scenario absent_local --strategy fedprox --mu 0.01
python -m src.scripts.run_client --node-id node3 --scenario absent_local --strategy fedprox --mu 0.01
```

**rare_expert with expert weighting:**

```bash
python -m src.scripts.run_server --strategy fedavg --expert-factor 2.0 --num-rounds 10
python -m src.scripts.run_client --node-id node1 --scenario rare_expert --strategy fedavg
python -m src.scripts.run_client --node-id node2 --scenario rare_expert --strategy fedavg
python -m src.scripts.run_client --node-id node3 --scenario rare_expert --strategy fedavg
```

### Step 4b — Run a registry-driven experiment

For structured ablations and comparable tracked runs, use the registry-based runner:

```bash
python -m src.scripts.run_experiment --experiment exp_v3_fedavg_normal_classweights
```

This runner:
- resolves a config bundle from `configs/experiment_registry.yaml`
- saves `resolved_config.json`
- writes `run_summary.json`, `round_metrics.json`, and `baseline_notes.md`
- logs the run to MLflow with a generated run name

> Note: the automation entrypoint currently targets `fedavg` and `fedprox` baseline-style runs. `scaffold` remains available through the manual server/client scripts while it is still under investigation.

### Step 5 — View MLflow results

```bash
mlflow ui --backend-store-uri ./outputs/mlruns --port 5000
# Open: http://localhost:5000
```

### Step 6 — Build the ablation table

```bash
python -m src.scripts.build_ablation_table
```

This exports:
- `outputs/reports/fl_v3_ablation_table.csv`
- `outputs/reports/fl_v3_ablation_table.md`

---

## 16. Repository Structure

```
fl-iot-ids-v3/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
├── VERSION
│
├── configs/
│   ├── experiment_registry.yaml ← named experiments for automation
│   ├── data/                    ← scenario bundles
│   ├── fl/                      ← FL strategy bundles
│   ├── imbalance/               ← class-weight / focal-loss bundles
│   ├── model/                   ← model bundles
│   ├── fl_config.yaml          ← FL hyperparameters (rounds, LR, batch size)
│   ├── global.yaml             ← Project name, scaler path
│   ├── model.yaml              ← Model architecture
│   └── nodes/
│       ├── node1.yaml
│       ├── node2.yaml
│       └── node3.yaml
│
├── src/
│   ├── common/
│   │   ├── config.py           ← YAML loader
│   │   ├── logger.py           ← Structured logging
│   │   ├── paths.py            ← Centralised path constants + scenario helpers
│   │   ├── schemas.py          ← Pydantic validation
│   │   └── utils.py            ← set_seed, misc helpers
│   │
│   ├── data/
│   │   ├── partitioning.py     ← Dirichlet partition
│   │   ├── preprocessor.py     ← Feature scaling
│   │   ├── dataset.py          ← PyTorch Dataset
│   │   ├── dataloader.py       ← DataLoader factory (80/20 split)
│   │   └── collector.py        ← Traffic collector (placeholder)
│   │
│   ├── model/
│   │   ├── network.py          ← MLPClassifier (28→128→64→34)
│   │   ├── train.py            ← train_one_epoch()
│   │   ├── evaluate.py         ← Evaluation helpers
│   │   └── losses.py           ← Weighted CE, Focal Loss
│   │
│   ├── fl/
│   │   ├── client_app.py       ← Flower ClientApp (alternative entry point)
│   │   ├── server_app.py       ← Flower ServerApp (alternative entry point)
│   │   ├── aggregation_hooks.py← Aggregated round metrics for tracked runs
│   │   ├── reporting_strategy.py ← FedAvg wrapper with artifact tracking
│   │   ├── strategy.py         ← Strategy definitions
│   │   └── metrics.py          ← Metric helpers
│   │
│   ├── scripts/
│   │   ├── run_experiment.py          ← Registry-driven experiment automation
│   │   ├── build_ablation_table.py    ← CSV/Markdown ablation summary builder
│   │   ├── fit_global_scaler.py        ← Fit StandardScaler on full dataset
│   │   ├── generate_scenarios.py       ← Partition + preprocess all scenarios
│   │   ├── generate_weights.py         ← Compute inverse-frequency class weights
│   │   ├── prepare_partitions.py       ← Low-level partition utility
│   │   ├── preprocess_node_data.py     ← Apply global scaler to one node
│   │   ├── run_server.py               ← FL server entry point
│   │   ├── run_client.py               ← FL client entry point
│   │   ├── validate_data_pipeline.py   ← Data sanity checks
│   │   ├── smoke_test.py               ← End-to-end smoke test
│   │   ├── test_dataloader.py          ← DataLoader smoke test
│   │   └── test_local_training.py      ← Single-node training test
│   │
│   ├── services/
│   │   ├── fl_client_service.py
│   │   ├── preprocessor_service.py
│   │   └── collector_service.py
│   │
│   ├── tracking/
│   │   ├── artifact_logger.py   ← Structured run artifacts for experiment baselines
│   │   └── run_naming.py        ← MLflow naming helpers
│   │
│   └── utils/
│       └── mlflow_logger.py            ← MLflowRunLogger (try/finally safe)
│
├── artifacts/                          ← Preprocessing outputs (not committed)
│   ├── scaler_standard_global.pkl      ← GlobalStandardScaler
│   ├── class_weights_34.pkl            ← Inverse-frequency weights
│   ├── feature_names.pkl               ← Feature name list
│   ├── scaffold_c_global.pkl           ← SCAFFOLD global control variate
│   └── scaffold_c_local_{node}.pkl     ← SCAFFOLD per-node control variates
│
├── data/                               ← Dataset (not committed)
│   ├── raw/{scenario}/{node}/train.csv
│   ├── processed/{scenario}/{node}/train_preprocessed.npz
│   └── splits/{scenario}_manifest.json
│
├── outputs/
│   ├── mlruns/                         ← MLflow tracking store
│   ├── logs/                           ← Per-script log files
│   ├── checkpoints/                    ← Model checkpoints
│   ├── metrics/                        ← JSON round metrics
│   └── reports/                        ← HTML visualisation reports
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_preprocessor.py
│   └── test_fl_smoke.py
│
└── deployments/docker/
    ├── base.Dockerfile
    ├── server.Dockerfile
    ├── client.Dockerfile
    └── docker-compose.yml
```

---

## References

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017.

2. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys 2020.

3. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML 2020.

4. Nour, M., Sharafaldin, I., & Ghorbani, A. A. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environments*. Sensors, 23(13), 5941.

5. Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Parcollet, T., & Lane, N. D. (2022). *Flower: A Friendly Federated Learning Research Framework*. arXiv:2007.14390.

---

**Last updated:** April 2026  
**Version:** v3 — Advanced FL Research Platform  
**Status:** Experimental — FedAvg and FedProx validated; SCAFFOLD under investigation
