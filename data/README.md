# Data Directory

The public GitHub repository does not store full raw or processed datasets.

Kaggle dataset: `COMING_SOON`

The planned public dataset package is a processed CICIoT2023-derived package for:

- preprocessing reproduction
- L1 binary IDS training
- federated learning partition reproduction
- QGA-selected feature analysis

Phase 3 local staging exists for the developer at:

```text
../qi-fl-ids-iot-kaggle-dataset/phase3_20260612_140955/
```

This is not a public URL.

Only tiny curated samples, placeholders, and documentation should be tracked in GitHub.

Do not commit:

- raw CICIoT2023 files
- processed `.csv`, `.parquet`, `.npz`, or `.npy` files
- model checkpoints
- scalers or pickle files
- packet captures
- credentials

Users must cite the original CICIoT2023 dataset:

```text
Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A.,
Lu, R., and Ghorbani, A. A. (2023).
CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks
in IoT Environments. Sensors, 23(13), 5941.
DOI: 10.3390/s23135941
```
