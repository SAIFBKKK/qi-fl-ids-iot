# Security Publication Checklist

This checklist tracks sensitive or local-only files that must be reviewed before public release. Do not print or copy secret values into issues, docs, commits, or release notes.

| Path | Risk | Public release decision | Recommended action | Policy |
| --- | --- | --- | --- | --- |
| `.claude/settings.local.json` | Local tool state may contain personal workflow settings or local paths. | Do not publish. | Moved to ignored `_private_local_only/.claude/settings.local.json`; keep out of Git. | Remove from Git; ignore private copy. |
| `.vscode/settings.json` | IDE settings may expose local paths or non-portable assumptions. | Do not publish in the public release. | File is untracked and ignored; leave local copy outside Git. | Ignore. |
| `data/fl.pcap` | Packet capture may contain network metadata or traffic evidence. | Do not publish in GitHub. | File is untracked and ignored; keep local only or review for private evidence storage. | Ignore/manual review. |
| `services/.env` | Runtime secrets and local service values. | Never publish. | File is untracked and ignored; keep only `services/.env.example` with placeholders. | Ignore. |
| `services/mosquitto/passwords` | MQTT password file. | Never publish. | File is untracked and ignored; regenerate locally when needed. | Ignore. |
| `services/scripts/generate_mqtt_password.sh` | Password-generation helper script. | Keep. | Reviewed as a generic helper with no obvious embedded secret patterns; keep tracked and document local usage with placeholders. | Keep reviewed helper. |

Before public release:

```powershell
git status --short
git ls-files | Select-String -Pattern '\.env|kaggle\.json|passwords|\.pem|\.key'
```

If any real secret appears, stop and rotate the credential outside Git.
