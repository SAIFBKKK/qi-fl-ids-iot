# Documentation Plan - QI-FL-IDS-IoT

## Documentation Goal

The public repository should let a new user understand, install, run, reproduce, and cite the framework without needing to read generated experiment folders. Documentation should be concentrated in important directories only.

## Current Documentation Findings

- The root `README.md` is polished but outdated: it frames QGA/QIFA/FedTN as future roadmap items, while final evidence selects `FedAvg + QGA` with 12 features.
- Documentation is split across root `docs/`, `experiments/qi-fl-ids-iot-final/docs/`, `experiments/*/outputs/reports/`, and service READMEs.
- Many P-phase reports are valuable as internal evidence but too granular for public first-contact documentation.
- Some docs contain absolute Windows paths and private lab IPs; sanitize before public release.
- There are many service-level README files. Keep these only where they help a user operate a service.

## README Files to Keep or Create

| File | Purpose | Target user | Key sections | Include | Avoid |
| --- | --- | --- | --- | --- | --- |
| `README.md` | Public entry point | New user, evaluator, maintainer | Short description, badges, features, architecture, repo structure, install, dataset, quick start, training, FL, QI modules, deployment, results, artifacts, docs, citation, license | Final selected model, links to Kaggle/artifacts placeholders, concise results, reproducibility contract | Long internal phase logs, generated report dumps, private paths/IPs |
| `docs/README.md` | Documentation index | Reader navigating docs | Doc map, recommended reading order, artifact/dataset links | Stable docs only | Every generated report |
| `src/README.md` or `experiments/qi-fl-ids-iot-final/src/README.md` | Code architecture guide | Developer | Package layout, module responsibilities, import conventions, extension points | `data`, `fl`, `qga`, `qifa`, `fedtn`, `deployment` module map | Tutorial content better suited to docs |
| `configs/README.md` or final stack config README | Reproducibility config guide | Experiment runner | Config hierarchy, key parameters, final model config, scenario parameters | Safe sample commands | Huge parameter dumps |
| `experiments/README.md` | Experiment inventory | Research reader | Which experiment folders are retained, legacy status, final stack path, output policy | Keep/legacy table | Detailed results tables already in docs |
| `services/README.md` | Microservice operations | MLOps/deployment user | Compose services, ports, env vars, health checks, MQTT topics, monitoring | Sanitized `.env.example`, demo workflow | Real secrets or private lab evidence |
| `notebooks/README.md` | Only if notebooks are kept | Research reader | Notebook policy, how to run, cleared-output requirement | Curated notebooks only | Old drafts and executed output-heavy notebooks |
| `assets/README.md` | Only if assets are kept | Maintainer | Which images are source assets vs generated figures | Small diagrams/logos | Generated result figure dump |

## Proposed `docs/` Structure

```text
docs/
  README.md
  01_overview.md
  02_architecture.md
  03_dataset.md
  04_preprocessing.md
  05_training.md
  06_federated_learning.md
  07_quantum_inspired_modules.md
  08_deployment.md
  09_live_lab.md
  10_results.md
  11_reproducibility.md
  12_troubleshooting.md
  references.md
```

## Document Content Plan

### `01_overview.md`

- Problem statement: IoT/WSN intrusion detection with privacy-preserving FL.
- Final project contribution: `FedAvg + QGA`, 12-feature IDS, Flower, Docker/MQTT live lab.
- What is quantum-inspired in this project.
- What is production-ready vs research evidence.

### `02_architecture.md`

- Offline pipeline architecture.
- Federated training architecture.
- Final deployment architecture: MQTT bridge, IDS API, dashboard, Prometheus/Grafana.
- Data and trust boundaries.
- Suggested Mermaid diagrams.

### `03_dataset.md`

- CICIoT2023 provenance and citation.
- Public Kaggle package contents.
- Original raw dataset vs processed balanced export vs final splits.
- Label mappings, feature names, QGA selected features.
- Licensing and limitations.

### `04_preprocessing.md`

- Input files and expected schema.
- Feature set: original 28 features and selected 12 QGA features.
- Scaling strategy and leakage prevention.
- Train/val/test split contract.
- Reproduction commands.

### `05_training.md`

- Centralized L1 baseline.
- Final L1 binary model.
- Hidden layers and threshold.
- Evaluation metrics and decision criteria.
- Where outputs are written locally.

### `06_federated_learning.md`

- Flower runtime.
- FedAvg baseline and final FedAvg + QGA.
- Non-IID partitioning strategy.
- Client/server responsibilities.
- Lightweight commands for smoke and full runs.

### `07_quantum_inspired_modules.md`

- QGA feature selection: search objective, final mask, validation-only selection.
- QIFA adaptive aggregation: scope and status.
- FedTN/MPS compression: scope and status.
- Distinguish validated final model from research modules.

### `08_deployment.md`

- Docker Compose profiles.
- Final IDS API.
- MQTT bridge.
- Online validator.
- Monitoring stack.
- Environment variables and `.env.example`.

### `09_live_lab.md`

- Safe live lab scope.
- MQTT topics.
- Traffic replay vs real passive observation.
- VM/network guidance with sanitized examples.
- What evidence should be externalized.

### `10_results.md`

- Final selected model metrics.
- Feature reduction summary.
- Comparison against baseline, QIFA, FedTN/MPS if retained.
- Link to external artifact pack for full figures/reports.

### `11_reproducibility.md`

- Python version, package installation, seeds, expected dataset layout.
- Small smoke test path.
- Full experiment path requiring Kaggle/external artifacts.
- Checksums and manifests.

### `12_troubleshooting.md`

- Missing dataset/artifact errors.
- Flower startup issues.
- Docker/MQTT issues.
- Windows path issues.
- MLflow/local output cleanup.

## Documentation Consolidation Rules

- Keep root docs stable and user-facing.
- Move P-phase implementation notes into an internal archive or external artifact pack unless they are essential to understand the final framework.
- Avoid README files in every tiny package; use package docstrings and `docs/02_architecture.md` instead.
- Keep service READMEs where operationally useful.
- Convert generated reports into one curated `docs/10_results.md`.
- Use placeholders for external URLs until Kaggle/cloud uploads are complete.

## README Draft Outline

```text
# QI-FL-IDS-IoT
Short description
Badges

## Key Features
## Final Selected Model
## Architecture
## Repository Structure
## Installation
## Dataset
## Quick Start
## Training Pipeline
## Federated Learning Pipeline
## Quantum-Inspired Modules
## Deployment and Live Lab
## Results
## External Artifacts
## Documentation
## Reproducibility
## Citation
## License
## Author and Acknowledgements
```

## Documentation Not to Add

- Do not add README files to every module directory.
- Do not document generated `outputs/`, `mlruns/`, caches, or temporary folders inside GitHub.
- Do not include large report exports in the main README.
- Do not expose real `.env` values, local usernames, or lab-only IP addresses as mandatory defaults.
