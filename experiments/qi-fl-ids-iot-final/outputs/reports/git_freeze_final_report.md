# Git Freeze Report — Quantum-Inspired FL IoT IDS Final

**Branch:** `final/quantum-inspired-fl-iot-ids-final`
**Date:** 2026-05-10
**Author:** Boukhatem

---

## Commits inclus dans ce gel

| Hash | Message |
|------|---------|
| `2cb2d92e` | feat(final): add reproducible quantum-inspired FL IDS pipeline P1-P9 |
| `42714390` | docs(final): add final experiment reports and ablation summaries P1-P9 |
| `b6aa9c2b` | docs(final): add final comparison figures for QGA and QIFA |
| *(ce commit)* | docs(final): add git freeze report |

---

## Tags prévus

| Tag | Phase |
|-----|-------|
| `final-v0.9-qga-l1-flower-final` | P8 QGA L1 Flower |
| `final-v0.9b-qga-l2-flower-final` | P8b QGA L2 Flower |
| `final-v0.10-qifa-l1-flower-final` | P9 QIFA L1 Flower |
| `final-v1.0-quantum-inspired-fl-iot-ids-final` | Release finale complète |

---

## Résumé des phases validées

| Phase | Description | Statut |
|-------|-------------|--------|
| P1 | Validation des données (CIC-IoT-2023) | ✅ code-ready |
| P2 | Préprocessing L1 binaire + L2 famille | ✅ code-ready |
| P3 | Partitionnement Dirichlet α ∈ {0.1, 0.5, 5.0}, K ∈ {3,4,5} | ✅ code-ready |
| P4 | Baseline centralisé L1 (MLP 28→128→64→2) | ✅ code-ready |
| P5 | FedAvg L1 grid 9 scénarios + Flower runtime | ✅ code-ready |
| P6 | FL hiérarchique L2/L3 Flower | ✅ code-ready |
| P7 | Multitier HeteroFL | ✅ code-ready |
| P8 | QGA Feature Selection L1 Flower + calibration conservative_seed_42 | ✅ code-ready |
| P8b | QGA Feature Selection L2 Flower | ✅ code-ready |
| P9 | QIFA Aggregation Flower (full_features=28 + qga_mask=12) | ✅ code-ready |

---

## Artefacts non versionnés (volontairement ignorés)

Les éléments suivants sont exclus du contrôle de version pour des raisons de taille ou de reproductibilité :

- **`outputs/preprocessed/`** — fichiers `.npz` scalés (reproductibles via `02_preprocess.py`)
- **`outputs/partitions/`** — partitions Dirichlet `.npz` (reproductibles via `03_dirichlet_split.py`)
- **`outputs/**/checkpoints/*.pth`** — poids PyTorch (reproductibles via les scripts d'entraînement)
- **`outputs/**/runs/`** — artefacts détaillés des runs Flower (smoke tests) : CSV métriques, logs
- **`outputs/fl_l1_fedavg/`**, **`outputs/fl_l1_flower/`** — résultats intermédiaires par run
- **`outputs/qga_feature_selection/calibration/`** — runs de calibration par profil/seed
- **`outputs/qifa_l1/`** — runs Flower QIFA individuels
- **`outputs/**/logs/*.log`** — logs Flower server/client

Ce qui est versionnés à la place : rapports agrégés dans `outputs/reports/`, masques finaux dans `outputs/qga*/final_selected_mask/`, figures de synthèse dans `outputs/figures/`.

---

## Commandes pour reproduire les rapports finaux

```bash
# Depuis la racine du repo
cd /chemin/vers/qi-fl-ids-iot

# P5 — Agréger résultats grid FedAvg
python experiments/qi-fl-ids-iot-final/src/scripts/05_3_aggregate_fl_grid_results.py \
  --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml

# P8 — Rapport ablation QGA L1
python experiments/qi-fl-ids-iot-final/src/scripts/08_build_qga_ablation_report.py \
  --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml

# P8b — Rapport ablation QGA L2
python experiments/qi-fl-ids-iot-final/src/scripts/08_b_build_qga_l2_ablation_report.py \
  --config experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml

# P9 — Smoke QIFA seul (28 features)
python experiments/qi-fl-ids-iot-final/src/scripts/09_run_qifa_flower_smoke.py \
  --config experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml \
  --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 \
  --variant hybrid --gamma 0.5 --address 127.0.0.1:8085

# P9 — Smoke QIFA + QGA (12 features conservative_seed_42)
python experiments/qi-fl-ids-iot-final/src/scripts/09_run_qifa_flower_smoke.py \
  --config experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml \
  --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 \
  --variant hybrid --gamma 0.5 --address 127.0.0.1:8086 --use-qga-mask

# P9 — Rapport ablation QIFA
python experiments/qi-fl-ids-iot-final/src/scripts/09_build_qifa_ablation_report.py \
  --config experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml
```

---

## Notes P10

P10 (Dashboard / FedTN / Déploiement) n'est pas inclus dans ce gel.
La branche `final/quantum-inspired-fl-iot-ids-final` est le point de départ stable pour P10.

Pour démarrer P10, créer une branche dédiée depuis ce tag :
```bash
git switch -c feat/p10-dashboard final-v1.0-quantum-inspired-fl-iot-ids-final
```
