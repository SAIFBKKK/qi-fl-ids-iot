# QGA Feature Selection

The QGA feature-selection module is a quantum-inspired optimiser for binary
feature masks in `experiments/fl-iot-ids-v3`. It does not modify raw CSV files
and does not refit scalers.

## Files

- Module: `experiments/fl-iot-ids-v3/src/qi/feature_selection.py`
- CLI: `experiments/fl-iot-ids-v3/src/scripts/run_qi_feature_selection.py`
- Config: `experiments/fl-iot-ids-v3/configs/qi/qga_feature_selection.yaml`
- Reduced model config: `experiments/fl-iot-ids-v3/configs/model/flat_34_qga_k16.yaml`

## Method

Each candidate solution is a binary vector of length `n_features`. A bit set to
1 means the feature is selected. The current v3 pipeline uses 28 features, so
`K=16` means selecting 16 out of 28 features.

The smoke/full selector uses:

- deterministic population initialisation
- exact-K mask repair
- crossover
- mutation
- train/validation filter fitness
- redundancy penalty for correlated selected features

The fitness is intentionally lightweight so it can run before expensive FL
training. The benchmark stage must still compare all-features and reduced
features with real FL runs.

## Artifacts

For a scenario such as `normal_noniid`, artifacts are written to:

```text
experiments/fl-iot-ids-v3/artifacts/qi_feature_selection/normal_noniid/
```

Expected files:

- `selected_features.json`
- `feature_mask.npy`
- `selection_report.md`

The `.npy` mask is ignored by Git policy. The JSON and Markdown files are small
and document the selected subset.

## Runtime Integration

The reduced model config enables feature selection:

```yaml
feature_selection:
  enabled: true
  method: qga
  k_features: 16
  artifact_path: artifacts/qi_feature_selection/{scenario}/selected_features.json
```

At client startup, the dataset loader applies `selected_indices` from the JSON
artifact to both train and validation NPZ files. The model input dimension is
then inferred from the reduced tensor shape.
