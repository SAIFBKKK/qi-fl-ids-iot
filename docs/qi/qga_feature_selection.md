# QGA Feature Selection

The QGA/QI feature-selection module is a quantum-inspired selector for the
28-feature CICIoT2023 representation used by `fl-iot-ids-v3`. It never edits raw
CSVs and never refits scalers.

## Files

- Selector: `experiments/fl-iot-ids-v3/src/qi/qi_feature_selector.py`
- CLI: `experiments/fl-iot-ids-v3/src/scripts/run_qi_feature_selection.py`
- Config: `experiments/fl-iot-ids-v3/configs/qi/qga_feature_selection.yaml`
- Reduced model: `experiments/fl-iot-ids-v3/configs/model/flat_28_qga_k15.yaml`

## Representation

The selector maintains a theta vector of length 28:

```text
theta = [theta_1, ..., theta_28]
p_i = sin(theta_i)^2
```

Each candidate mask is sampled from `p_i`, then repaired to exactly `K`
selected features. For the sprint benchmark, `K=15`.

After each generation, theta moves toward the best mask:

```text
theta <- theta + alpha * (theta_best - theta)
```

This is quantum-inspired search over binary masks. It is not quantum hardware.

## Fitness

Fitness is based on a mini `MLPClassifier`:

- sample up to `max_samples_per_class`
- split internally into train/validation
- train a small MLP for `epochs`
- score validation Macro-F1
- optionally apply a light feature-count penalty

The smoke mode keeps `n_generations`, `pop_size`, `epochs` and samples small.
The full mode is configured but should be run only when time is available.

## Artifacts

For `normal_noniid`, artifacts are saved under:

```text
experiments/fl-iot-ids-v3/artifacts/qi_feature_selection/normal_noniid/
```

Expected files:

- `selected_features.json`
- `feature_mask.npy`
- `selection_report.md`

The `.npy` mask is ignored by Git. The JSON and Markdown files are small and
document the selected subset.

## Runtime Integration

Reduced-feature experiments use:

```yaml
feature_selection:
  enabled: true
  method: qga
  k_features: 15
  artifact_path: artifacts/qi_feature_selection/{scenario}/selected_features.json
```

At client startup, `selected_indices` are applied to train and validation NPZ
files before the MLP is built. The model therefore receives 15 inputs while the
source data remains the validated 28-feature representation.
