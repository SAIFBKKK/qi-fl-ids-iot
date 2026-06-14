# Preprocessing

Preprocessing prepares CICIoT2023-derived data for intrusion detection and federated partitioning.

Typical responsibilities:

- load processed or raw-compatible CICIoT2023 inputs
- clean invalid values
- map labels for L1 binary IDS and L2 family analysis
- scale numerical features
- create train/validation/test splits
- generate federated partitions
- record feature names, label mappings, and validation summaries

The public repository keeps only small metadata and code. Large `.csv`, `.parquet`, `.npz`, `.npy`, `.pkl`, and scaler artifacts are externalized.

Use `<REPO_ROOT>` as the checkout root in local instructions. Restore dataset files from the Kaggle package only when running full experiments.
