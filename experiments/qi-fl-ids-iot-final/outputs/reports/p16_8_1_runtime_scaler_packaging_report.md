# P16.8.1 Live Lab Runtime Scaler Packaging

## Objective

P16.8.1 packages the final L1 binary RobustScaler into a lightweight JSON runtime artifact that can be cloned by the live lab VMs.

## Observed Problem

During P16.8 VM runs, the realtime agent reported:

```text
Scaler file not found
Scaler unavailable; returning unscaled 28-feature vector.
```

The MQTT-to-IDS pipeline worked, but VM-generated features were not actually scaled.

## Solution

Export the local final scaler pickle to:

`experiments/live_lab/artifacts/l1_binary_robust_scaler.json`

The JSON contains the scaler type, 28-feature order, `center`, and `scale`. The VM runtime applies the RobustScaler-compatible transform:

`scaled = (x - center) / scale`

## Why The Pickle Is Not Committed

The pickle remains an experiment output artifact and is not suitable for lightweight VM packaging. The live lab Git clone should carry only small, transparent runtime artifacts.

## Modes Affected

- `selected_12_scaled`: load JSON scaler, scale 28 extracted features, then apply QGA mask `conservative_seed_42`.
- `original_28_scaled`: load JSON scaler and emit 28 scaled features.

Expected dry-run runtime after P16.8.1:

- `scaler.available = true`
- `scaler.used = true`
- `scaler.source = json`

## Limits

- The JSON is a runtime packaging artifact, not a model update.
- P16.8.1 does not change P8-P16 results.
- P16.8.1 does not run packet capture, training, Flower, or lab scenarios.
