# Phase 3 Externalization Report

## Scope

Phase 3 moved heavy generated artifacts and dataset files outside the GitHub working tree while preserving traceability with manifests and checksums.

No permanent deletion, upload, Git history rewrite, training, or source-code refactor was performed.

## Branch

- Working branch: `cleanup/phase3-external-artifacts`
- Source branch before Phase 3: `final/quantum-inspired-fl-iot-ids-final`

## External Locations

Kaggle staging folder:

```text
C:\Users\saifb\dev\qi-fl-ids-iot-kaggle-dataset\phase3_20260612_140955
```

External artifact staging folder:

```text
C:\Users\saifb\dev\qi-fl-ids-iot-external-artifacts\phase3_20260612_140955
```

Release manifest mirror:

```text
C:\Users\saifb\dev\qi-fl-ids-iot-release-manifests\phase3_20260612_140955
```

Compressed external artifact archive:

```text
C:\Users\saifb\dev\qi-fl-ids-iot-external-artifacts\qi-fl-ids-iot-artifacts-v1-20260612.zip
```

Archive SHA256:

```text
582163c484d070aa7dbf3d8600254465513158bed75f8012d8094aaa8813342d
```

## Kaggle Staging

- Files moved by `StageKaggleDataset`: `238`
- Moved size: `6525.924 MB`
- Final staged folder files including README, citation, metadata, manifests, and checksums: `244`
- Final staged folder size: `6526.205 MB`

Staged for Kaggle:

- `data/balancing_v3_fixed300k_outputs/`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/`
- selected QGA/preprocessing metadata under generated final outputs when classified as dataset metadata

Kaggle sidecar files created:

- `README.md`
- `CITATION.md`
- `dataset-metadata.json`
- `MANIFEST.csv`
- `MANIFEST.json`
- `checksums.sha256`

No model checkpoints, runtime scalers, MLflow runs, logs, private IP evidence, `.env`, password files, packet captures, or Kaggle credentials were included in the Kaggle package.

## External Artifacts

- Files moved by `MoveExternalArtifacts`: `12092`
- Moved size: `9039.032 MB`
- Final external staging folder files including README, manifests, and checksums: `12096`
- Final external staging folder size: `9054.388 MB`
- Compressed archive size: `4694.007 MB`

Moved to external artifact storage:

- generated experiment outputs from `experiments/qi-fl-ids-iot-final/outputs/`
- generated experiment outputs from `experiments/fl-iot-ids-v3/outputs/`
- root `outputs/`
- MLflow runs
- logs
- generated reports and figures
- model checkpoints and deployment binaries
- scalers and preprocessing binaries
- non-Kaggle heavy arrays and archives
- legacy/generated dataset splits and processed artifacts

The first `Compress-Archive` attempt failed with an OutOfMemory exception and produced no archive. The script was updated to use streaming `System.IO.Compression.ZipArchive`, then compression succeeded.

## Manifests And Checksums

Canonical repo-side manifests:

```text
phase3_manifests/
```

Important manifest files:

- `phase3_manifests/phase3_session.json`
- `phase3_manifests/20260612_140955_phase3_plan.csv`
- `phase3_manifests/20260612_140955_phase3_plan.json`
- `phase3_manifests/20260612_141032_kaggle_staging.csv`
- `phase3_manifests/20260612_141032_kaggle_staging.json`
- `phase3_manifests/20260612_141457_external_artifacts.csv`
- `phase3_manifests/20260612_141457_external_artifacts.json`
- `phase3_manifests/20260612_142207_archive.summary.json`
- `phase3_manifests/20260612_142547_verify.summary.json`

External checksum files:

- Kaggle package: `checksums.sha256`
- External artifact staging folder: `checksums.sha256`
- External artifact archive: `qi-fl-ids-iot-artifacts-v1-20260612.zip.sha256`
- Release manifest mirror: `checksums.sha256`

## Intentionally Kept In GitHub

- source code
- configs
- docs
- tests
- scripts
- Dockerfiles and docker-compose files
- README files
- `.env.example` files
- tiny curated demo subsets under `data/cic-iot-2023/demo_subsets/`
- selected small metadata outside generated output folders, such as final deployment manifests and schemas
- curated documentation assets under `docs/`

## Manual Review Items Not Moved

The script intentionally left these in place:

- `.claude/settings.local.json` - local tool state
- `.vscode/settings.json` - local IDE state
- `data/fl.pcap` - packet capture
- `services/.env` - local environment file
- `services/mosquitto/passwords` - local password file
- `services/scripts/generate_mqtt_password.sh` - suspicious secret/password-named helper requiring manual review

No secret values were printed.

## `.gitignore` Changes

The `.gitignore` was tightened so generated reports and figures under `outputs/` and `experiments/**/outputs/` remain ignored by default.

Removed broad exceptions that previously re-allowed:

- `outputs/reports/*.html`
- `outputs/reports/*.md`
- `outputs/reports/*.csv`
- `outputs/reports/*.json`
- `experiments/**/outputs/reports/*.html`
- `experiments/**/outputs/reports/*.md`
- `experiments/**/outputs/reports/*.csv`
- `experiments/**/outputs/reports/*.json`
- `experiments/fl-iot-ids-v3/outputs/reports/qi_benchmark_reduced/**`
- `experiments/**/figures/*.png`
- `experiments/**/figures/*.svg`

Kept exceptions for curated docs/assets and small selected metadata.

## Verification

Final script verification:

- remaining Kaggle candidates: `0`
- remaining external artifact candidates: `0`
- manual review candidates: `6`
- `kaggle.json` found in repo: `0`

Git tracked-file count:

- `git ls-files | Measure-Object`: `2749`

Current Git status summary:

- Modified files: `3`
- Deleted tracked artifact files: `1701`
- Untracked entries: `4`
- Total status entries: `1708`

The large number of tracked deletions is expected because previously tracked generated outputs, reports, figures, metadata exports, and artifact files were moved out of GitHub into external staging.

The ignored-file sample from `git ls-files -o -i --exclude-standard` starts with `.venv/`, confirming ignored local environment files remain ignored.

## Remaining Risks

- Manual-review files listed above still need policy decisions before public release.
- Some tracked generated artifacts were previously committed; this phase removes them from the working tree but does not rewrite Git history.
- The `.git/` directory remains large until a separate history cleanup decision is made.
- External artifacts and Kaggle package have not been uploaded.
- Dataset license and derived Kaggle package license must be verified before upload.

## Next Recommended Phase

Phase 4 should focus on public repository hardening:

- review and commit Phase 3 changes
- manually resolve the six manual-review items
- update the root README with `COMING_SOON` links
- add or update `CITATION.cff`
- run lightweight CI/import tests with tiny samples only
- prepare Kaggle/cloud upload instructions without uploading yet
