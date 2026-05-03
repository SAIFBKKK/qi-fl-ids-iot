# Rapport Git — Pré-Sprint 0
**Date :** 2026-05-02 | **Heure de génération :** début Sprint 0

## Branche courante
```
feat/microservices
```
Synchronisée avec `origin/feat/microservices` — aucun commit en avance/retard.

## Fichiers modifiés / untracked
**Aucun** — `git status --porcelain` retourne vide. Le worktree est propre.

> Note : La gitStatus initiale de la session montrait des fichiers M/? (docs/reports/MODEL_FACTORY_30ROUNDS_REPORT.md, run_status.json, model_factory_summary.json, dossier weak/). Ces fichiers ont été inclus dans le commit `90dea0dd feat: add model factory and dynamic Mode A tier assignment` — visible dans `git log`. Le worktree est donc propre au démarrage du sprint.

## Stashes existants
```
stash@{0}: On feat/multitier-fl: wip: US7 exported_models partial
```

### Recommandation sur le stash
| Stash | Branche d'origine | Contenu probable | Recommandation | Justification |
|---|---|---|---|---|
| stash@{0} | `feat/multitier-fl` | WIP US7 exported_models (partiel) | **(b) Garder stashé** pour l'instant | La branche `feat/multitier-fl` est distincte de `feat/microservices`. Ce WIP pourrait être pertinent si le travail US7 doit être mergé, mais il ne bloque pas le Sprint 0. À revoir après le sprint. |

## État Model Factory (info pour la suite)
Le dossier `experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/` est commité et complet :
- `weak/` : global_model.pth, scaler.pkl, feature_names.pkl, label_mapping.json, model_config.json ✅
- `medium/` : idem ✅
- `powerful/` : idem ✅
- `deployment_data/` : deployment_15.parquet, feature_names.pkl, label_mapping.json, split_summary.json ✅

## Verdict
**Aucune action requise avant de démarrer le sprint.** Le worktree est propre, les bundles sont en place.
