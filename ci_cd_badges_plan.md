# CI/CD and Badges Plan - QI-FL-IDS-IoT

## Current Workflow Audit

Detected workflow:

```text
.github/workflows/ci.yml
```

Current jobs:

- `lint-and-test-v2`
- `lint-and-test-v3`
- `lint-and-test-v1`

Current triggers:

```yaml
push:
  branches: [main, develop]
pull_request:
  branches: [main]
```

## CI Risks

| Risk | Evidence | Impact |
| --- | --- | --- |
| Final branch not covered | Current branch is `final/quantum-inspired-fl-iot-ids-final`; workflow triggers only `main` and `develop` for push | CI badge may not reflect final branch health. |
| Final authoritative stack not covered | Workflow does not run `experiments/qi-fl-ids-iot-final` tests | Public framework could publish without testing final model/deployment code. |
| Legacy jobs may break public CI | v1/v2 are historical and may rely on stale dependencies | Public CI should not be blocked by legacy stacks unless maintained. |
| Dataset/artifact-dependent tests | `experiments/qi-fl-ids-iot-final/tests` reference `data/balancing_v3_fixed300k_outputs`, `.npz`, `outputs/reports`, and checkpoints | CI will fail on clean clone unless these tests are skipped or marked. |
| Heavy dependency installs | `environment.yml` includes GPU/quantum/Ray dependencies; v3 requirements include MLflow/Flower/PyTorch | CI time and reliability risk. |
| Badge link too generic | README badge points to workflow badge but links to repository actions page | Prefer workflow-specific link. |
| License mismatch | Root `LICENSE` is MIT; `experiments/fl-iot-ids-v3/pyproject.toml` says Apache-2.0 | License badge may be misleading until aligned. |

## Lightweight CI Plan

Recommended workflows:

```text
.github/workflows/ci.yml
.github/workflows/docker-smoke.yml
.github/workflows/docs.yml
```

### `ci.yml`

Purpose: fast source-quality gate.

Jobs:

- Python setup: 3.11, optionally 3.12 after dependencies are verified.
- Install minimal dev dependencies from a root `pyproject.toml` or `requirements-dev.txt`.
- `ruff check`.
- Import smoke tests for final package/modules.
- Unit tests that use only synthetic data.
- Exclude tests marked `slow`, `requires_dataset`, `requires_artifacts`, `requires_docker`, `requires_live_lab`.

Suggested test command:

```powershell
python -m pytest -m "not slow and not requires_dataset and not requires_artifacts and not requires_docker and not requires_live_lab" --tb=short
```

### `docker-smoke.yml`

Purpose: prove Dockerfiles build without running heavy training.

Jobs:

- Build final IDS API Docker image.
- Build MQTT bridge Docker image.
- Build dashboard or service stack image.
- Optionally run `docker compose config` only.

Avoid:

- Full Flower training.
- Pulling large datasets.
- Long-running MQTT/live lab demos.

### `docs.yml`

Purpose: public documentation hygiene.

Jobs:

- Markdown lint if configured.
- Link check for internal relative links.
- Optional external link check with retries.
- Ensure README mentions dataset/artifact placeholders.

## Test Marker Plan

Add or enforce pytest markers:

```ini
[pytest]
markers =
    slow: long-running tests
    requires_dataset: needs Kaggle/CICIoT2023 data
    requires_artifacts: needs external artifacts or generated outputs
    requires_docker: needs Docker daemon
    requires_live_lab: needs MQTT/live lab environment
```

Move artifact-dependent tests out of default CI:

- `experiments/qi-fl-ids-iot-final/tests/data/*`
- integration tests requiring `outputs/preprocessed`, `outputs/partitions`, `outputs/reports`, or checkpoints
- service tests that need `global_model.pth` unless a tiny fixture model is generated in test setup

## Badge Plan for README

Recommended badges:

```markdown
[![CI](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/ci.yml)
[![Docker Smoke](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docker-smoke.yml/badge.svg?branch=main)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docker-smoke.yml)
[![Docs](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/SAIFBKKK/qi-fl-ids-iot/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.x-FF6B6B.svg)](https://flower.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20TBD-20BEFF.svg)](#dataset)
[![Artifacts](https://img.shields.io/badge/Artifacts-External%20TBD-6f42c1.svg)](#external-artifacts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Framework-orange.svg)](#project-status)
```

Do not add a Kaggle badge with a fake URL. Use `TBD` until uploaded.

## README Badge Fixes

- Replace generic CI link with workflow-specific URL.
- Choose one status: recommended `Research Framework` or `Final Year Project - Public Release`.
- Add Kaggle dataset placeholder badge after dataset staging.
- Add external artifacts placeholder badge after artifact pack staging.
- Align license badge with actual root license.
- Update Python badge only if CI actually tests that version.
- Avoid claiming Docker build health until Docker smoke workflow exists.

## Recommended Public CI Policy

- Default CI must pass without datasets, MLflow artifacts, model checkpoints, Docker daemon, MQTT broker, or internet downloads beyond pip.
- Full reproducibility checks should be documented as local commands, not mandatory pull-request CI.
- Artifact-dependent tests can run in a manual workflow after downloading Kaggle/external artifacts.
- Docker smoke can be optional or path-filtered to `services/**`, `deployment/**`, and Dockerfiles.

## Later Workflow Outline

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: ruff check .
      - run: python -m pytest -m "not slow and not requires_dataset and not requires_artifacts and not requires_docker and not requires_live_lab" --tb=short
```

This outline assumes a root `pyproject.toml` exists. If not, use final stack-specific install commands.
