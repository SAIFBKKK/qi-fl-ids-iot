# Data Links

## Sommaire

Ce dossier ne doit pas contenir de copie du dataset. Les phases suivantes utiliseront des chemins relatifs ou des symlinks controles vers `../../data/balancing_v3_fixed300k_outputs/`, en particulier `balancing_v3_fixed300k_balanced.parquet` et `label_mapping.json`.

Commandes de verification des chemins, sans creation de donnees :

PowerShell Windows :

```powershell
Get-ChildItem ..\..\data\balancing_v3_fixed300k_outputs
```

Linux/macOS bash :

```bash
ls -lh ../../data/balancing_v3_fixed300k_outputs
```

[Phase P1 - À implémenter]
