# L1 Final Deployment Bundle

This bundle packages the final production L1 IDS model selected by the project evidence pack:

- Model: P8 FedAvg + QGA
- Task: L1 binary IDS
- Selected mask: `conservative_seed_42`
- Input dimension: 12 selected scaled features
- Labels: `normal=0`, `attack=1`

The bundle intentionally excludes training data, test data, partitions, logs, and Flower run directories.

## Build

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_build_delivery_manifest.py
```

If the local P8 checkpoint is present and below 5 MB, it is copied to:

```text
artifacts/model.pth
```

The API can still start without the checkpoint, but `/ready` will report `false`.

## Files

- `deployment_manifest.json`: bundle status and artifact hashes
- `selected_model.json`: production model metadata
- `model_registry_deployment.json`: deployment model registry
- `qga_mask_reference.json`: final QGA mask references
- `feature_schema.json`: original and selected feature schema
- `metrics_reference.json`: P13/P12 metrics used for reporting
- `artifacts/`: packaged lightweight artifacts only

The API expects scaled features. It accepts either 12 selected scaled features or 28 original scaled features and applies the QGA mask internally.
