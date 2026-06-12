# External Artifacts

Heavy generated artifacts were externalized during Phase 3 so the GitHub repository can stay readable and lightweight.

External artifacts archive: COMING_SOON

The local Phase 3 staging folder is outside the repository:

```text
../qi-fl-ids-iot-external-artifacts/phase3_20260612_140955/
```

The compressed local archive is:

```text
../qi-fl-ids-iot-external-artifacts/qi-fl-ids-iot-artifacts-v1-20260612.zip
```

The archive contains generated or runtime-specific material such as:

- model checkpoints and deployment binaries
- scalers, encoders, and preprocessing binaries
- MLflow runs
- generated logs, figures, reports, and experiment outputs
- legacy raw experiment splits and intermediate artifacts

These files are not required to inspect the source code. They are useful for reproducing exact reported results or restoring the live-demo/deployment state.

## Restore Workflow

After downloading the artifact archive, extract it at the repository root so original relative paths are restored.

```powershell
Expand-Archive -Path qi-fl-ids-iot-artifacts-v1-20260612.zip -DestinationPath .
```

Then verify checksums with the archive-provided `checksums.sha256` file.

## Public Release Notes

- Do not upload `.env` files, password files, private keys, Kaggle credentials, or private lab inventories.
- Review live-lab logs and reports before publication because they may include local paths or private IP addresses.
- Keep only curated documentation assets under `docs/` in GitHub.
