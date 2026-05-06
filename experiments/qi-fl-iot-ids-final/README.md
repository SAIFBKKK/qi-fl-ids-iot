# Quantum-Inspired Federated Learning IoT IDS — Final

## Sommaire

## Statut courant

Phase P0 (Audit + Skeleton) terminée.

Ce repertoire consolide la validation finale du PFE autour d'un IDS IoT en apprentissage federe avec modules quantum-inspired. Il repart du dataset CIC-IoT-2023 equilibre deja present dans le depot, des generations experimentales `v1`, `v2`, `v3`, et des microservices existants. P0 ne contient aucune logique metier nouvelle : uniquement le squelette, les placeholders et la carte de reutilisation.

## Decisions gelees

- Dataset : `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet` (`9 401 350 x 29`).
- Espace d'entree : 28 features + 1 `label_id`.
- Splits non-IID cibles : Dirichlet alpha dans `{0.1, 0.5, 5.0}`.
- Nombre de clients cible : `K` dans `{3, 4, 5}`.
- L1 binaire : production et dashboard final.
- L2/L3 : experimentaux, rapport seulement.
- Multi-tier : weak / medium / powerful avec poids partages type HeteroFL.
- Quantum-inspired : QGA + QIFA + FedTN/MPS sur tier powerful uniquement.
- QIARM : reporte en perspective.
- PowerSGD : non retenu, FedTN/MPS direct.

## Pipeline cible

Data validation → Preprocessing → Dirichlet split → Centralized L1 → FL L1 baseline → L2/L3 experimental → Multi-tier HeteroFL → QGA → QIFA + attacks → FedTN/MPS → Ablation → Dashboard L1 → Docker stack → Rapport final.

## Roadmap

| Phase | Nom | Statut |
| --- | --- | --- |
| P0 | Audit + Skeleton | fait |
| P1 | Data validation | à venir |
| P2 | Preprocessing train-only | à venir |
| P3 | Dirichlet split | à venir |
| P4 | Centralized L1 baseline | à venir |
| P5 | FL L1 baseline | à venir |
| P6 | Hierarchical L2/L3 experimental | à venir |
| P7 | Multi-tier HeteroFL | à venir |
| P8 | QGA feature selection | à venir |
| P9 | QIFA aggregation | à venir |
| P10 | Robustness and attacks | à venir |
| P11 | FedTN/MPS compression | à venir |
| P12 | Ablation and evaluation reports | à venir |
| P13 | Dashboard L1 final | à venir |
| P14 | Docker stack and final delivery | à venir |

## Comment executer

À venir, phase par phase après validation utilisateur.

Commandes de verification P0 documentees :

PowerShell Windows :

```powershell
Get-ChildItem -Recurse experiments\qi-fl-iot-ids-final
git status --short
```

Linux/macOS bash :

```bash
find experiments/qi-fl-iot-ids-final -maxdepth 5 -type f | sort
git status --short
```

[Phase P0 - Audit + Skeleton]
