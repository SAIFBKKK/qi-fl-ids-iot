# External Artifacts

This directory is a placeholder for artifacts that are useful for reproduction, reporting, or demonstrations but should not be stored directly in GitHub.

The following artifact types belong outside GitHub:

- Model checkpoints and deployment bundles.
- Scalers, encoders, class weights, and preprocessing binaries.
- Generated figures and plots.
- Training, Flower, Docker, MQTT, and live-lab logs.
- Generated reports and experiment exports.
- MLflow runs and tracking metadata.
- Large validation outputs and raw experiment evidence.

External artifacts archive: COMING_SOON

During Phase 3, artifacts were staged locally outside the repository:

```text
../qi-fl-ids-iot-external-artifacts/phase3_20260612_140955/
../qi-fl-ids-iot-external-artifacts/qi-fl-ids-iot-artifacts-v1-20260612.zip
```

Download links will be added after the archive is reviewed and uploaded to external storage.

Artifacts are versioned separately from the source repository. Each published artifact bundle should include a manifest with original paths, file sizes, checksums, and the Git commit used to produce it.

Do not place secrets, private keys, raw credentials, or private live-lab inventories in artifact bundles.

To restore artifacts later, extract the archive at the repository root so original relative paths are recreated.
