# Dataset

Kaggle dataset: COMING_SOON

The public GitHub repository does not store the full processed dataset. Heavy processed files were staged outside the repository during Phase 3:

```text
../qi-fl-ids-iot-kaggle-dataset/phase3_20260612_140955/
```

This staged package is intended for a future Kaggle upload and preserves original repository-relative paths.

## Contents

The staged dataset package contains:

- balanced CICIoT2023-derived files from `data/balancing_v3_fixed300k_outputs/`
- final preprocessed L1/L2 data from `experiments/qi-fl-ids-iot-final/outputs/preprocessed/`
- federated partition files from `experiments/qi-fl-ids-iot-final/outputs/partitions/`
- small metadata and manifests required for traceability
- `README.md`, `CITATION.md`, `dataset-metadata.json`, `MANIFEST.csv`, `MANIFEST.json`, and `checksums.sha256`

## Intended Use

Use the Kaggle package for:

- preprocessing reproduction
- L1 binary IDS training
- federated learning partition reproduction
- QGA-selected feature analysis

Tiny demo subsets may remain in GitHub only when they are required for examples, smoke tests, or documentation.

## Citation

Users must cite the original CICIoT2023 dataset and paper. The license of the derived Kaggle package must be verified before public upload.

The final Kaggle link and exact citation metadata will be added after publication.
