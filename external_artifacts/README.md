# External Artifacts

This directory is a GitHub placeholder only.

External artifacts archive: `COMING_SOON`

Heavy generated artifacts were externalized during Phase 3 and should not be committed back into GitHub.

Local developer archive name:

```text
qi-fl-ids-iot-artifacts-v1-20260612.zip
```

Archive SHA256:

```text
582163c484d070aa7dbf3d8600254465513158bed75f8012d8094aaa8813342d
```

Artifacts include:

- model checkpoints and deployment binaries
- scalers, encoders, and preprocessing binaries
- generated figures and reports
- training, Flower, Docker, MQTT, and live-lab logs
- MLflow runs and tracking metadata
- deployment bundles

Restore locally only:

```powershell
Expand-Archive -Path <ARTIFACTS_DIR>/qi-fl-ids-iot-artifacts-v1-20260612.zip -DestinationPath <REPO_ROOT>
```

Verify before extraction:

```powershell
Get-FileHash <ARTIFACTS_DIR>/qi-fl-ids-iot-artifacts-v1-20260612.zip -Algorithm SHA256
```

Do not place secrets, private keys, `.env` files, MQTT password files, Kaggle credentials, packet captures, or private lab inventories in artifact bundles.
