# Multi-tier Federated Learning Branch

## Branch

`feat/multitier-fl`

## Stable Parent

This branch starts from the stable baseline snapshot:

```bash
git checkout baseline-fl-v1
```

The tag `baseline-fl-v1` preserves the FedAvg baseline before Multi-tier FL changes begin.

## Purpose

`feat/multitier-fl` isolates the Multi-tier Federated Learning work from the validated US1 baseline. The goal is to support heterogeneous IoT nodes by assigning model capacity and training settings according to each node profile.

Node profiles may include CPU cores, RAM, latency, battery status, device type, and network quality. The server will use those signals to assign tiers such as weak, medium, or powerful.

## Planned Scope

### US3 - Node Profiler and Tier Assignment

Clients report resource profiles to the server.

```json
{
  "node_id": "node_01",
  "cpu_cores": 2,
  "ram_mb": 1024,
  "device_type": "raspberry_pi_zero",
  "avg_latency_ms": 120,
  "battery_powered": true,
  "network_quality": "low"
}
```

The server assigns a tier and may adapt local training parameters.

```json
{
  "assigned_tier": "weak",
  "model_width": 0.25,
  "local_epochs": 1,
  "batch_size": 128
}
```

### US4 - SuperNet and Sub-models

Define a shared SuperNet and derive tier-specific subnetworks.

```text
Weak     : 28 -> 64 -> 34
Medium   : 28 -> 128 -> 64 -> 34
Powerful : 28 -> 256 -> 128 -> 34
```

### US5 - Masked Aggregation

Aggregate compatible parameter slices across models with different capacities.

### US6 - Multi-tier FL Experiment

Run and compare the Multi-tier FL experiment against the stable FedAvg baseline.

### US7 - Packaging and Reporting

Export tier-specific models, metrics, logs, and deployment artifacts.

## Return to Baseline

To inspect the stable baseline:

```bash
git checkout baseline-fl-v1
```

To return to Multi-tier development:

```bash
git checkout feat/multitier-fl
```

## Development Rules

- Keep baseline corrections separate from Multi-tier implementation commits.
- If the baseline needs a fix, apply it on `main`, retag if appropriate, then merge or cherry-pick into this branch.
- Run baseline smoke tests before and after major Multi-tier changes.
- Keep US3-US7 commits scoped so regressions are easy to isolate.

## Status (April 26, 2026)

Multi-tier Static HeteroFL implementation **frozen at US6**.

### Completed (US1-US6)

- Baseline FL bundle exported and validated
- Branch `feat/multitier-fl` with isolated changes
- Node Profiler with static `tier_profiles.yaml`
- SuperNet with 3 sub-models (weak/medium/powerful)
- Masked aggregation (Static HeteroFL)
- Validation experiment with 3 runs (baseline / control / multitier)

### Frozen (US7+)

The following extensions were designed but not implemented:

- **US7**: Per-tier model export (3 separate bundles)
- **Tier-aware inference deployment**: would require US7 bundles
- **FedRolex** (rolling sub-model extraction): perspective only

### Rationale for freezing

The Multi-tier core algorithm is validated. The remaining sprint capacity
(approximately 13 working days) was reallocated to:

- Microservices architecture (US10-12)
- Quantum-Inspired modules (QGA, QIARM)
- Demo + documentation

### Inference path going forward

The microservices infrastructure (US10+) will use the **baseline
MLPClassifier bundle** (`outputs/deployment/baseline_fedavg_normal_classweights/`)
produced in US1. Tier-aware deployment with heterogeneous sub-models
is documented as Phase 5 perspective.
