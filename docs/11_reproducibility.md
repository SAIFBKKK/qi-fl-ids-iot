# Reproducibility

The repository is designed so source code and lightweight documentation remain public while heavy data and generated artifacts are restored only when needed.

Reproduction levels:

1. **Code review**: clone the repository and inspect source/docs.
2. **Lightweight tests**: run unit or import checks that use tiny samples.
3. **Dataset reproduction**: restore the Kaggle dataset package when available.
4. **Exact artifact reproduction**: restore the external artifacts archive.

Dataset placeholder:

```text
Kaggle dataset: COMING_SOON
```

Artifacts placeholder:

```text
External artifacts archive: COMING_SOON
```

Use manifests and checksums to verify restored files.

Do not restore heavy artifacts into a public Git branch.
