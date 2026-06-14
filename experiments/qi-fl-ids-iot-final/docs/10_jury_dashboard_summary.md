# P10 — Jury & Dashboard Summary

## Message simple pour le jury

Ce projet développe un système de détection d'intrusions (IDS) pour objets connectés (IoT)
utilisant l'**apprentissage fédéré** combiné à des algorithmes d'optimisation **quantique-inspirés**.

**Problème** : Les données IoT sont distribuées sur des appareils hétérogènes (non-IID).
Un modèle centralisé est impossible (données privées). Le fédéré classique gaspille de la bande passante.

**Solution apportée** :
- **QGA** (Quantum Genetic Algorithm) : sélectionne automatiquement les features les plus discriminantes (28 → 12), réduisant la bande passante de 17%.
- **QIFA** (Quantum-Inspired Federated Aggregation) : pondère intelligemment les clients selon leur qualité, réduisant les fausses alarmes (FPR −21%).

---

## Modèle recommandé — Production L1

**P8 FedAvg + QGA** (`conservative_seed_42`, 12 features)

| Métrique | Valeur | vs Baseline FL |
|---|---|---|
| Macro F1 | **0.9480** | +0.73 pts |
| Attack Recall | **0.9550** | +0.76 pts |
| FPR | 0.0594 | −0.69 pts |
| Bande passante | 7.24 MB | −17% |
| Taille modèle | 40 200 octets | −17% |
| Features | **12 / 28** | −57% |
| True Flower runtime | ✅ | — |

---

## Alternatives selon objectif

### Faible FPR (peu de fausses alarmes) → P9 QIFA seul
- FPR = **0.0524** (−21% vs P5 FedAvg)
- Macro F1 = 0.9454
- 28 features, 8.71 MB bande passante

### Meilleur Attack Recall → P9 QIFA + QGA
- Attack Recall = **0.9592** (meilleur FL)
- Macro F1 = 0.9471
- 12 features, 7.24 MB bande passante

### Classification famille L2 (expérimental) → P8-b QGA L2
- 19 features (vs 28)
- Macro F1 = 0.6466 (+1.1 pts vs P6 baseline)
- True Flower runtime

---

## Métriques clés à afficher dans le dashboard

```
┌────────────────────────────────────────────────────────────┐
│  MODÈLE ACTIF : P8 FedAvg + QGA (L1 Binary)               │
│  Features : 12 / 28 (conservative_seed_42)                 │
├──────────────────┬─────────────────────────────────────────┤
│  Macro F1        │  0.9480  ████████████████████░ 94.8%    │
│  Attack Recall   │  0.9550  █████████████████████░ 95.5%   │
│  FPR             │  0.0594  ██░░░░░░░░░░░░░░░░░░░  5.9%   │
│  Accuracy        │  0.9481  █████████████████████░ 94.8%   │
├──────────────────┼─────────────────────────────────────────┤
│  Bande passante  │  7.24 MB / 30 rounds                    │
│  Rounds          │  30 (Flower true runtime)               │
│  Clients         │  3 (α=0.5)                              │
└──────────────────┴─────────────────────────────────────────┘
```

### Comparaison rapide pour le jury

| Méthode | F1 | Recall | FPR | BW | Features |
|---|---|---|---|---|---|
| P4 Centralized (référence) | 0.9611 | 0.9526 | 0.0293 | — | 28 |
| P5 FedAvg | 0.9407 | 0.9474 | 0.0663 | 8.71 MB | 28 |
| **P8 FedAvg+QGA** ← prod | **0.9480** | **0.9550** | 0.0594 | **7.24 MB** | **12** |
| P9 QIFA | 0.9454 | 0.9436 | **0.0524** | 8.71 MB | 28 |
| P9 QIFA+QGA | 0.9471 | **0.9592** | 0.0658 | **7.24 MB** | **12** |

---

## Architecture finale simplifiée

```
IoT Devices (K=3 clients)
    │ Données locales (partitions Dirichlet α=0.5)
    │ Features sélectionnées par QGA : 12/28
    │
    ▼
Flower Server (Aggregation)
    ├─ FedAvg classique (P5/P8)
    │    Poids ∝ nb_samples
    │
    └─ QIFA (P9)
         Amplitudes quantiques : aᵢ = cos(θᵢ/2)
         Probabilités : pᵢ = aᵢ²
         Poids final : wᵢ = γ·pᵢ + (1−γ)·fedavg_wᵢ   [γ=0.5]
    │
    ▼
Global Model (MLP: input→128→64→2)
    ├─ P8+QGA : input=12 → taille 40 200 octets
    └─ P9 QIFA : input=28 → taille 48 392 octets

Évaluation finale sur global test holdout (jamais vu pendant training)
```

---

## Ce qui est en dehors du scope de ce projet

- Déploiement microservices (Docker / API REST) → P11+
- Dashboard temps réel → P11+
- FedTN (Transfer Learning fédéré) → P11+
- Robustness grid complète P9 → non lancée (smoke tests validés)
