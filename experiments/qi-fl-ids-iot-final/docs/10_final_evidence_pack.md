# P10 — Final Evidence Pack

## Résumé du pipeline P4 → P9

Ce document synthétise les résultats de toutes les phases du projet
**Quantum-Inspired Federated Learning pour IDS IoT** sur le dataset CIC-IoT-2023.

---

## Architecture générale

```
CIC-IoT-2023 (34 classes, 46 features bruts)
    ↓ P2: Preprocessing L1 binaire (Normal/Attack) + L2 famille (12 familles)
    ↓ P3: Partitionnement Dirichlet (α ∈ {0.1, 0.5, 5.0}, K ∈ {3,4,5})
    ↓ P4: Baseline centralisé MLP 28→128→64→2
    ↓ P5: FedAvg L1 — 9 scénarios
    ↓ P6: FL Hiérarchique L2 (Flower true runtime)
    ↓ P7: Multi-tier HeteroFL (L1+L2)
    ↓ P8: QGA Feature Selection L1 → conservative_seed_42 (12/28 features)
    ↓ P8-b: QGA Feature Selection L2 (19/28 features)
    ↓ P9: QIFA Aggregation (quantum-inspired weighted FedAvg)
```

Global test holdout séparé dès P2 — jamais utilisé pour la sélection de modèle ou de seuil.

---

## Tableau des meilleurs résultats L1 binaire (α=0.5, K=3)

| Phase | Method | Features | Macro F1 | Attack Recall | FPR | Accuracy | BW (MB) |
|---|---|---|---|---|---|---|---|
| P4 | Centralized L1 | 28 | **0.9611** | 0.9526 | 0.0293 | 0.9612 | 0 |
| P5 | FedAvg L1 | 28 | 0.9407 | 0.9474 | 0.0663 | 0.9409 | 8.71 |
| P8 | FedAvg + QGA L1 | **12** | **0.9480** | **0.9550** | 0.0594 | 0.9481 | **7.24** |
| P9 | QIFA L1 | 28 | 0.9454 | 0.9436 | **0.0524** | 0.9455 | 8.71 |
| P9 | QIFA + QGA L1 | **12** | 0.9471 | **0.9592** | 0.0658 | 0.9473 | **7.24** |

**Meilleur Macro F1 FL** : P8 FedAvg + QGA (0.9480)
**Meilleur Attack Recall** : P9 QIFA + QGA (0.9592)
**Meilleur FPR** : P9 QIFA (0.0524 — −21% vs P5)
**Meilleure compression** : P8 et P9+QGA (57% de réduction features)

---

## Tableau des résultats L2 famille

| Phase | Method | Features | Macro F1 | Accuracy | BW (MB) |
|---|---|---|---|---|---|
| P6 | L2 Flower baseline | 28 | 0.6355 | 0.6841 | 8.99 |
| P8-b | L2 FedAvg + QGA | 19 | 0.6466 | 0.7555 | 8.16 |
| P7 | Multi-tier L1 best (α=5.0) | 28 | 0.9490 | 0.9490 | 13.12 |
| P7 | Multi-tier L2 best (α=0.5) | 28 | 0.7085 | 0.8194 | 13.49 |

L2 est une tâche plus difficile (12 classes de famille). P8-b améliore P6 de +1.1 pts F1 avec 32% moins de features.

---

## Comparaison QGA vs QIFA

| Critère | QGA (P8) | QIFA (P9) | QIFA+QGA (P9) |
|---|---|---|---|
| Objectif | Sélection features | Agrégation FL adaptative | Les deux |
| Features | 12 (conservative) | 28 (toutes) | 12 |
| Macro F1 | 0.9480 | 0.9454 | 0.9471 |
| Attack Recall | 0.9550 | 0.9436 | **0.9592** |
| FPR | 0.0594 | **0.0524** | 0.0658 |
| BW (MB) | **7.24** | 8.71 | **7.24** |
| True Flower | ✅ | ✅ | ✅ |

QGA et QIFA sont orthogonaux et complémentaires. QIFA seul réduit le FPR ; QGA réduit la bande passante et améliore le recall ; combinés ils maximisent le recall.

---

## Meilleur modèle production recommandé

**P8 FedAvg + QGA L1** (`conservative_seed_42`, 12 features) :
- Macro F1 = 0.9480 (meilleur parmi les méthodes FL)
- Bande passante réduite de 17% vs P5 FedAvg
- Modèle 17% plus léger (40 200 vs 48 392 octets)
- True Flower runtime validé, accepted=True

---

## Alternatives selon objectif

| Objectif | Méthode recommandée |
|---|---|
| Meilleure détection d'attaques | P9 QIFA + QGA (recall=0.9592) |
| Moins de fausses alarmes | P9 QIFA seul (FPR=0.0524) |
| Compression maximale | P8 / P9+QGA (12 features) |
| Classification famille L2 | P8-b L2 QGA (19 features) |
| Référence centralisée | P4 Centralized L1 (F1=0.9611) |

---

## Limites scientifiques

1. **Smoke tests uniquement pour P9** : QIFA validé sur 1 round × 1000 samples, pas 30 rounds complets.
2. **P7 Multi-tier** : résultats expérimentaux, non finalisés en production.
3. **Non-IID extrême** (α=0.1) : toutes les méthodes dégradent, particulièrement K=5.
4. **L2 famille** : classification multi-classe difficile, macro_f1 < 0.71 même au meilleur scénario.
5. **QGA calibration** : `conservative_seed_42` sélectionné par engineering score sur short validations — non garanti optimal sur données de production.

---

## Tags Git utilisés

| Tag | Phase |
|---|---|
| `final-v0.9-qga-l1-flower-final` | P8 QGA L1 |
| `final-v0.9b-qga-l2-flower-final` | P8-b QGA L2 |
| `final-v0.10-qifa-l1-flower-final` | P9 QIFA L1 |
| `final-v1.0-quantum-inspired-fl-iot-ids-final` | Release P1-P9 |
| `final-v1.1-evidence-pack` | P10 Evidence Pack |

---

## Commandes de reproduction principales

```bash
# P8 — Smoke QGA Flower L1
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_fedavg_flower_smoke.py \
  --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml \
  --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000

# P9 — Smoke QIFA seul
python experiments/qi-fl-ids-iot-final/src/scripts/09_run_qifa_flower_smoke.py \
  --config experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml \
  --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 \
  --variant hybrid --gamma 0.5 --address 127.0.0.1:8085

# P9 — Smoke QIFA + QGA
python experiments/qi-fl-ids-iot-final/src/scripts/09_run_qifa_flower_smoke.py \
  --config experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml \
  --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 \
  --variant hybrid --gamma 0.5 --address 127.0.0.1:8086 --use-qga-mask

# P10 — Reconstruire l'evidence pack
python experiments/qi-fl-ids-iot-final/src/scripts/10_build_final_evidence_pack.py
```
