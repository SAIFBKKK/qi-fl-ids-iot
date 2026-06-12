# External Artifacts

External artifacts archive: `COMING_SOON`

Heavy generated artifacts were externalized during Phase 3 so the GitHub repository can remain focused on source, configs, docs, scripts, Docker files, tests, selected metadata, and tiny samples.

Local developer archive name:

```text
qi-fl-ids-iot-artifacts-v1-20260612.zip
```

Archive SHA256:

```text
582163c484d070aa7dbf3d8600254465513158bed75f8012d8094aaa8813342d
```

The public URL is not available yet.

Artifacts include:

- model checkpoints
- scalers and preprocessing binaries
- logs
- generated reports
- generated figures
- MLflow runs
- deployment bundles
- raw experiment evidence

## Restore Workflow

Restore only in a local working copy:

```powershell
Expand-Archive -Path <ARTIFACTS_DIR>/qi-fl-ids-iot-artifacts-v1-20260612.zip -DestinationPath <REPO_ROOT>
```

Verify the archive checksum before extraction:

```powershell
Get-FileHash <ARTIFACTS_DIR>/qi-fl-ids-iot-artifacts-v1-20260612.zip -Algorithm SHA256
```

Compare the result with:

```text
582163c484d070aa7dbf3d8600254465513158bed75f8012d8094aaa8813342d
```

Do not commit restored artifacts back into GitHub.
