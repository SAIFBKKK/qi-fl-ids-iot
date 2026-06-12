# Security Publication Checklist

This checklist tracks sensitive or local-only files that must be reviewed before public release. Do not print or copy secret values into issues, docs, commits, or release notes.

| Path | Risk | Public release decision | Recommended action | Policy |
| --- | --- | --- | --- | --- |
| `.claude/settings.local.json` | Local tool state may contain personal workflow settings or local paths. | Do not publish. | Remove from public branch or keep ignored locally. | Ignore/remove. |
| `.vscode/settings.json` | IDE settings may expose local paths or non-portable assumptions. | Do not publish unless sanitized and intentionally generic. | Prefer removal from public release; keep a separate `.vscode/extensions.json` only if needed later. | Ignore/remove. |
| `data/fl.pcap` | Packet capture may contain network metadata or traffic evidence. | Do not publish in GitHub. | Keep out of repository; if evidence is needed, review and place in private external artifacts. | Ignore/manual review. |
| `services/.env` | Runtime secrets and local service values. | Never publish. | Keep only `services/.env.example` with placeholders. | Ignore/remove from public branch. |
| `services/mosquitto/passwords` | MQTT password file. | Never publish. | Regenerate locally from documented steps; do not commit. | Ignore/remove from public branch. |
| `services/scripts/generate_mqtt_password.sh` | Name references password generation; may be safe source but needs review. | Keep only if script contains no secrets and is useful. | Review contents; sanitize examples; document usage with placeholders. | Review, then keep or sanitize. |

Before public release:

```powershell
git status --short
git ls-files | Select-String -Pattern '\.env|kaggle\.json|passwords|\.pem|\.key'
```

If any real secret appears, stop and rotate the credential outside Git.
