# External Artifacts Plan - QI-FL-IDS-IoT

## Goal

Create an external artifact package for files that are important for evidence, demos, or reproducibility, but inappropriate for GitHub and not necessarily part of the Kaggle dataset.

## Artifact Categories

| Category | Current examples | Destination | GitHub action later |
| --- | --- | --- | --- |
| Trained models | `*.pth`, `*.pt`, deployment `model.pth`, run checkpoints | `external_artifacts/models/` | Remove from Git unless tiny demo model is intentionally kept. |
| Scalers and preprocessing binaries | `scaler_*.pkl`, `class_weights_*.pkl`, `feature_names_*.pkl` | `external_artifacts/scalers/` and `external_artifacts/preprocessing/` | Replace with JSON metadata when possible. |
| QGA/QIFA/FedTN outputs | QGA masks, selection decisions, QIFA sweeps, FedTN reports | `external_artifacts/qi_modules/` | Keep only final selected JSON metadata in Git. |
| MLflow runs | `outputs/mlruns`, `experiments/*/outputs/mlruns` | `external_artifacts/mlflow/` | Remove from Git working tree and ignore. |
| Figures | `outputs/figures`, report figures, confusion matrices | `external_artifacts/figures/` | Keep only curated docs images. |
| Logs | `*.log`, runtime logs, Flower logs | `external_artifacts/logs/` | Remove from public Git. |
| Generated reports | P-phase `.md`, `.csv`, `.html`, `.json`, PDFs | `external_artifacts/reports/` | Consolidate final results into docs. |
| Deployment bundles | Final l1 bundle, model/scaler/mask packages | `external_artifacts/deployment_bundles/` | Keep bundle manifest in Git; externalize binary payload. |
| Notebooks archive | old notebooks and output-heavy notebooks | `external_artifacts/notebooks_archive/` | Keep only curated/cleared notebooks if needed. |
| Raw experiment outputs | `experiments/fl-iot-ids-v3/outputs`, `experiments/qi-fl-ids-iot-final/outputs` | `external_artifacts/raw_experiment_outputs/` | Remove or replace with manifests. |

## Proposed Folder Structure

```text
external_artifacts/
  README.md
  manifest.json
  checksums.sha256
  models/
    final_l1_fedavg_qga/
    baseline_models/
    legacy_checkpoints/
  scalers/
    l1_binary/
    l2_family/
  preprocessing/
    label_mappings/
    feature_names/
    class_weights/
  qi_modules/
    qga/
    qifa/
    fedtn_mps/
  figures/
    final_report/
    qga/
    qifa/
    robustness/
    live_lab/
  logs/
    flower/
    docker/
    mqtt/
    training/
  reports/
    final_curated/
    phase_reports/
    html_exports/
  mlflow/
    fl_v3/
    qi_final/
    root_outputs/
  deployment_bundles/
    l1_final/
    docker_runtime/
  notebooks_archive/
    baseline_ciciot2023/
    nsl_kdd_legacy/
  raw_experiment_outputs/
    fl_iot_ids_v1/
    fl_iot_ids_v2/
    fl_iot_ids_v3/
    qi_fl_ids_iot_final/
```

## Compression Plan

Recommended archive name:

```text
qi-fl-ids-iot-artifacts-v1.zip
```

Recommended contents:

- `external_artifacts/README.md`.
- `manifest.json` with original path, new path, file size, category, reason, and SHA256.
- All artifact files moved with relative paths preserved.
- `checksums.sha256`.

PowerShell compression command for later:

```powershell
Compress-Archive -Path external_artifacts\* -DestinationPath qi-fl-ids-iot-artifacts-v1.zip -CompressionLevel Optimal
Get-FileHash qi-fl-ids-iot-artifacts-v1.zip -Algorithm SHA256
```

For very large artifacts, prefer `7z` or cloud-native upload tooling and split the archive by category:

```text
qi-fl-ids-iot-models-v1.zip
qi-fl-ids-iot-reports-v1.zip
qi-fl-ids-iot-mlflow-v1.zip
qi-fl-ids-iot-live-lab-evidence-v1.zip
```

## External README Contents

`external_artifacts/README.md` should include:

- What the artifact pack contains.
- Which Git commit it corresponds to.
- How to verify checksums.
- How to restore artifacts into the repo for full reproduction.
- Which artifacts are optional.
- Which artifacts are required for final deployment demo.
- Privacy note for logs/live-lab evidence.

## Links to Add Later in GitHub README

Add placeholders first, then replace after upload:

```text
Dataset: [Kaggle dataset - coming soon](TBD)
Artifacts: [External artifact pack v1 - coming soon](TBD)
Final model bundle: [FedAvg + QGA L1 deployment bundle - coming soon](TBD)
Full experiment evidence: [Reports and figures archive - coming soon](TBD)
```

## Keep in GitHub vs External

Keep in GitHub:

- `selected_model.json`.
- `deployment_manifest.json`.
- `feature_schema.json`.
- `qga_mask_reference.json` if it contains only JSON metadata.
- Final selected features JSON if small and not derived from private data in a problematic way.
- Small README examples and config files.

Externalize:

- Model binaries unless explicitly kept as tiny demo artifacts.
- Scaler pickles.
- Checkpoints.
- Full `outputs/`.
- Full `mlruns/`.
- Full report and figure packs.
- Live lab evidence with IPs/logs.

## Verification Before Upload

```powershell
Get-ChildItem external_artifacts -Recurse -File |
  Group-Object Extension |
  Sort-Object Count -Descending |
  Select-Object Name, Count

Get-ChildItem external_artifacts -Recurse -File |
  Sort-Object Length -Descending |
  Select-Object -First 50 FullName, Length
```

Upload should happen only after secret scan and explicit approval.
