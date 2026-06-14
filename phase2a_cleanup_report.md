# Phase 2A Cleanup Report

## Scope

Phase 2A prepared the repository for public cleanup without deleting source code, moving datasets, moving model artifacts, rewriting Git history, uploading files, or running training.

Actions performed:

- Created a safety branch and tag at the pre-cleanup commit.
- Updated `.gitignore` for public-release safety.
- Added a dry-run-first cleanup and inventory script.
- Added/sanitized public-safe environment templates.
- Added public release helper documentation.
- Ran safe inventories and quarantined only reproducible cache/temp folders.

## Git Safety Snapshot

- Active branch: `final/quantum-inspired-fl-iot-ids-final`
- Backup branch created: `backup/pre-public-cleanup-20260612-124745`
- Safety tag created: `pre-public-cleanup-20260612-124745`
- Branch was not switched during this phase.

Initial preflight showed only prior audit markdown files as untracked. No unexpected modified source files were present before Phase 2A edits.

## Files Modified

- `.gitignore`
- `services/.env.example`
- `experiments/live_lab/configs/server.env.example`
- `data/README.md`

## Files Created

- `scripts/cleanup_repo_artifacts.ps1`
- `SECURITY.md`
- `external_artifacts/README.md`
- `phase2a_cleanup_report.md`

Generated inventory/quarantine files:

- `_cleanup_manifests/20260612-125301_inventory.csv`
- `_cleanup_manifests/20260612-125301_inventory.summary.json`
- `_cleanup_manifests/20260612-125308_cache_dry_run.csv`
- `_cleanup_manifests/20260612-125308_cache_dry_run.summary.json`
- `_cleanup_manifests/20260612-125328_cache_quarantine.csv`
- `_cleanup_manifests/20260612-125328_cache_quarantine.summary.json`
- `_cleanup_manifests/20260612-125340_heavy_inventory.csv`
- `_cleanup_manifests/20260612-125340_heavy_inventory.summary.json`
- `_cleanup_manifests/20260612-125355_secret_scan.csv`
- `_cleanup_manifests/20260612-125355_secret_scan.summary.json`
- `_cleanup_manifests/20260612-125603_cache_dry_run.csv`
- `_cleanup_manifests/20260612-125603_cache_dry_run.summary.json`
- `_cleanup_manifests/20260612-125834_cache_dry_run.csv`
- `_cleanup_manifests/20260612-125834_cache_dry_run.summary.json`
- `_cleanup_manifests/20260612-125914_cache_restore.csv`
- `_cleanup_manifests/20260612-125914_cache_restore.summary.json`

Cache quarantine destination:

- `_cleanup_quarantine/20260612-125328/`

## `.gitignore` Summary

The ignore rules now cover:

- Python caches and local virtual environments.
- Notebook checkpoints.
- IDE and OS metadata.
- `.env`, `.env.*`, Kaggle credentials, private keys, and credential-like files.
- Raw/processed datasets and packet captures.
- ML artifacts including `.npz`, `.npy`, `.pkl`, `.joblib`, `.pth`, `.pt`, `.onnx`, and `.ckpt`.
- Logs, generated reports, generated figures, MLflow runs, archives, and Docker local runtime state.
- `external_artifacts/*` while allowing `external_artifacts/README.md`.
- `_cleanup_quarantine/`.

The rules keep important public metadata and configuration visible:

- `.env.example`
- `README.md`
- `.gitkeep`
- YAML configs
- small selected-model metadata such as `selected_model.json`, `selected_features.json`, `feature_mask.json`, `deployment_manifest.json`, and `feature_schema.json`
- curated sample folders such as `data/samples/` and `tests/fixtures/`

## Cleanup Script

Created `scripts/cleanup_repo_artifacts.ps1` with the following modes:

```powershell
.\scripts\cleanup_repo_artifacts.ps1 -Mode Inventory
.\scripts\cleanup_repo_artifacts.ps1 -Mode CacheDryRun
.\scripts\cleanup_repo_artifacts.ps1 -Mode CacheQuarantine -Confirm
.\scripts\cleanup_repo_artifacts.ps1 -Mode HeavyInventory
.\scripts\cleanup_repo_artifacts.ps1 -Mode SecretScan
```

Safety properties:

- Default mode is inventory only.
- No deletion behavior exists.
- Cache quarantine refuses to run without `-Confirm`.
- Quarantine moves only cache/temp folders into `_cleanup_quarantine/<timestamp>/`.
- Relative paths are preserved in quarantine.
- `.git/`, `.venv/`, virtualenv folders, `node_modules/`, and existing quarantine folders are skipped.
- Heavy inventory is report-only.
- Secret scan reports only file paths and risk types, not secret values.
- The script now skips cache/temp candidates that contain tracked files, and treats source/report-like files under `tmp/` as unsafe.

PowerShell parse check passed after the script was updated to support zero-record manifests.

## Inventory Results

General inventory:

- Manifest: `_cleanup_manifests/20260612-125301_inventory.csv`
- Records: `28`
- Total inventoried size excluding skipped local/system folders: `15637.078 MB`

Cache dry-run before quarantine:

- Manifest: `_cleanup_manifests/20260612-125308_cache_dry_run.csv`
- Candidates: `97`
- Total size: `5.715 MB`
- Candidate types:
  - `__pycache__`: `86`
  - `.pytest_cache`: `6`
  - `.ruff_cache`: `4`
  - `tmp`: `1`

Cache quarantine:

- Performed: yes
- Manifest: `_cleanup_manifests/20260612-125328_cache_quarantine.csv`
- Initial quarantined candidates: `97`
- Initial total size: `5.715 MB`
- Destination: `_cleanup_quarantine/20260612-125328/`

Correction after final verification:

- Root `tmp/` contained tracked Markdown files and was restored from quarantine.
- Restore manifest: `_cleanup_manifests/20260612-125914_cache_restore.csv`
- Restored size: `0.108 MB`
- Effective cache-only directories remaining in quarantine: `96`
- Effective cache-only size remaining in quarantine: `5.608 MB`

Post-quarantine cache dry-run:

- Final manifest: `_cleanup_manifests/20260612-125834_cache_dry_run.csv`
- Candidates remaining outside quarantine: `1`
- Candidate status: `tmp` is `SKIP_UNSAFE` because it contains tracked files.
- Total skipped size: `0.108 MB`

## Heavy Artifact Inventory

- Manifest: `_cleanup_manifests/20260612-125340_heavy_inventory.csv`
- Candidates: `10860`
- Total size: `15528.582 MB`

Category summary:

| Category | Count | Total MB |
| --- | ---: | ---: |
| heavy_extension | 1624 | 15494.940 |
| generated_evidence | 840 | 32.280 |
| mlflow | 8396 | 1.360 |

Largest report-only candidates:

| Path | Size MB |
| --- | ---: |
| `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.csv` | 2285.768 |
| `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet` | 929.973 |
| `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/train_scaled.npz` | 753.401 |
| `experiments/fl-iot-ids-v3/data/raw/rare_expert/node3/train.csv` | 702.302 |
| `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/train.parquet` | 683.042 |

No heavy artifacts were moved in this phase.

## Sensitive-Risk Inventory

- Manifest: `_cleanup_manifests/20260612-125355_secret_scan.csv`
- Risk records: `8653`
- Total scanned candidate size represented in records: `2.53 MB`

Risk summary:

| Risk type | Count |
| --- | ---: |
| mlflow_metadata | 8396 |
| local_windows_path | 171 |
| private_lab_ip | 58 |
| possible_secret_assignment | 24 |
| password_file | 2 |
| environment_file | 1 |
| packet_capture | 1 |

High-priority manual review paths:

- `services/.env` - environment file and possible secret assignment.
- `services/mosquitto/passwords` - password file.
- `data/fl.pcap` - packet capture.
- `experiments/live_lab/**` - private lab IP references in configs, docs, scripts, and node profiles.
- `experiments/fl-iot-ids-v3/outputs/mlruns/**` - MLflow metadata.
- `experiments/fl-iot-ids-v3/outputs/reports/**` - generated reports with local Windows paths.
- `experiments/qi-fl-ids-iot-final/outputs/**` - generated outputs with local path metadata and heavy artifacts.

No secret values were printed or copied.

## Remaining Risks

- Large datasets and processed arrays remain in the working tree and must be staged for Kaggle or external storage in a later phase.
- MLflow runs and generated reports remain in place for now.
- `services/.env` and `services/mosquitto/passwords` remain local files and must never be committed.
- Live-lab configs/docs still contain private IP references that should be templated or documented before public release.
- Packet capture data remains local and should be excluded from GitHub.
- `.git/` remains large due repository history; history rewrite was intentionally not performed.

## Recommended Next Phase

Proceed to Phase 3: heavy artifact externalization and Kaggle staging.

Recommended objectives:

- Build a reviewed external artifact manifest.
- Stage processed dataset files for Kaggle packaging.
- Move model checkpoints, scalers, logs, figures, reports, MLflow runs, and deployment bundles into an external artifact folder.
- Generate checksums before compression.
- Replace local heavy artifacts with README links, manifests, or small metadata files.
