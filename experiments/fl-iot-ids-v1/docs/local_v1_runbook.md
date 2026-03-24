# Local V1 Runbook

## Table des matières
- [Local V1 Runbook](#local-v1-runbook)
  - [Table des matières](#table-des-matières)
  - [Objet](#objet)
  - [Pré-requis](#pré-requis)
  - [1. Préparation des partitions locales](#1-préparation-des-partitions-locales)
  - [2. Préprocessing local des nœuds](#2-préprocessing-local-des-nœuds)
  - [3. Vérification du DataLoader](#3-vérification-du-dataloader)
  - [4. Test d'entraînement local mono-client](#4-test-dentraînement-local-mono-client)
  - [5. Exécution locale Flower — mode explicite recommandé](#5-exécution-locale-flower--mode-explicite-recommandé)
  - [6. Exécution Flower moderne — optionnelle](#6-exécution-flower-moderne--optionnelle)
  - [7. Logs](#7-logs)
  - [8. Fichiers de sortie importants](#8-fichiers-de-sortie-importants)
  - [9. Commande de gel des dépendances](#9-commande-de-gel-des-dépendances)
  - [10. Statut](#10-statut)
  - [Notes pour Windows/PowerShell](#notes-pour-windowspowershell)
  - [Dépannage](#dépannage)
    - [ModuleNotFoundError](#modulenotfounderror)
    - [Données manquantes](#données-manquantes)
    - [Erreurs de preprocessing](#erreurs-de-preprocessing)
    - [Problèmes Flower](#problèmes-flower)

## Objet
Ce document décrit les commandes et étapes nécessaires pour exécuter localement la V1 du système FL IDS IoT.

## Pré-requis
- Environnement Python fonctionnel (Python 3.8+ recommandé)
- Dépendances installées (`pip install -r requirements.txt`)
- Artifacts baseline présents dans `artifacts/` (feature_names.pkl, scaler_robust.pkl, label_mapping_34.pkl)
- Données baseline disponibles (train.csv du dataset CIC IoT 2023)
- Projet exécuté depuis la racine du repo `fl-iot-ids-v1`
- Variables d'environnement : `PYTHONPATH=.` pour les imports locaux

## 1. Préparation des partitions locales
Cette étape crée les partitions locales pour `node1`, `node2` et `node3` en divisant stratifiquement les données d'entraînement.

```bash
python -m src.scripts.prepare_partitions
```

**Résultat attendu :**
- `data/raw/node1/train.csv`
- `data/raw/node2/train.csv`
- `data/raw/node3/train.csv`
- `data/splits/partition_manifest.json`

**Note :** Nécessite l'accès au dataset baseline dans `../baseline-CIC_IOT_2023/raw/train/train.csv`

## 2. Préprocessing local des nœuds

Cette étape applique le preprocessing baseline-compatible à chaque nœud (normalisation, encodage des labels, etc.).

**Node 1**
```bash
python -m src.scripts.preprocess_node_data --node-id node1
```

**Node 2**
```bash
python -m src.scripts.preprocess_node_data --node-id node2
```

**Node 3**
```bash
python -m src.scripts.preprocess_node_data --node-id node3
```

**Résultat attendu :**
- `data/processed/node1/train_preprocessed.npz`
- `data/processed/node2/train_preprocessed.npz`
- `data/processed/node3/train_preprocessed.npz`

**Note :** Chaque fichier NPZ contient les arrays X (features) et y (labels) prétraités.

## 3. Vérification du DataLoader

Test rapide du chargement PyTorch des données prétraitées.

```bash
python -m src.scripts.test_dataloader
```

**Résultat attendu :**
- Chargement du dataset OK
- Nombre de batches affiché (ex: ~10000+ batches)
- Shape des tensors correcte (batch: [256, 33], labels: [256])

## 4. Test d'entraînement local mono-client

Validation du modèle local sans Flower (entraînement sur un seul nœud).

```bash
python -m src.scripts.test_local_training
```

**Résultat attendu :**
- Initialisation du modèle OK
- Entraînement local OK
- Loss en baisse progressive
- Accuracy en progression

## 5. Exécution locale Flower — mode explicite recommandé

Ce mode a servi de validation principale de la V1 locale. Utilise 4 terminaux séparés.

**Terminal 1 — Serveur**
```bash
python -m src.scripts.run_server --host 127.0.0.1 --port 8080 --num-rounds 3 --min-clients 3
```

**Terminal 2 — Client node1**
```bash
python -m src.scripts.run_client --node-id node1 --server-address 127.0.0.1:8080 --local-epochs 1
```

**Terminal 3 — Client node2**
```bash
python -m src.scripts.run_client --node-id node2 --server-address 127.0.0.1:8080 --local-epochs 1
```

**Terminal 4 — Client node3**
```bash
python -m src.scripts.run_client --node-id node3 --server-address 127.0.0.1:8080 --local-epochs 1
```

**Résultat attendu :**
- Le serveur échantillonne 3 clients sur 3 disponibles
- `aggregate_fit` reçoit 3 résultats par round
- `aggregate_evaluate` reçoit 3 résultats par round
- 3 rounds s'exécutent sans échec
- La loss distribuée baisse
- L'accuracy distribuée augmente

## 6. Exécution Flower moderne — optionnelle

Cette validation utilise l'interface moderne de Flower.

```bash
flwr run . local-simulation
```

Ou avec configuration personnalisée :
```bash
flwr run . local-simulation --run-config "num-server-rounds=3 local-epochs=1 batch-size=256 learning-rate=0.001"
```

**Remarque :**
Sous Windows, ce mode peut produire :
- Des warnings Ray
- Du bruit de logs
- Des messages protobuf

Le fonctionnement global peut rester correct malgré cela.

## 7. Logs

Les logs applicatifs sont sauvegardés dans `outputs/logs/`.

**Exemples :**
- `outputs/logs/fl_server.log`
- `outputs/logs/fl_client.log`
- `outputs/logs/preprocessor.log`

## 8. Fichiers de sortie importants

- `data/splits/partition_manifest.json` (métadonnées des partitions)
- `data/processed/node1/train_preprocessed.npz` (données prétraitées node1)
- `data/processed/node2/train_preprocessed.npz` (données prétraitées node2)
- `data/processed/node3/train_preprocessed.npz` (données prétraitées node3)
- `outputs/logs/` (logs détaillés)
- `outputs/checkpoints/` (modèles sauvegardés)

## 9. Commande de gel des dépendances

Pour figer l'environnement local validé :

```bash
python -m pip freeze > requirements-lock.txt
```

## 10. Statut

La V1 locale est considérée comme validée et prête pour la phase suivante :
- Dockerisation
- Docker-compose
- Préparation au déploiement dans un environnement Linux/WSL2 plus stable

## Notes pour Windows/PowerShell

- Utilisez `python -m` au lieu de `python` direct pour les scripts
- Définissez `PYTHONPATH=.` avant l'exécution : `$env:PYTHONPATH = "."`
- Pour les commandes multi-lignes, utilisez le backtick `` ` `` pour la continuation
- Évitez les backslashes `\` comme continuation de ligne (bash-style)

## Dépannage

### ModuleNotFoundError
- Vérifiez que vous êtes dans le répertoire racine du projet
- Définissez `PYTHONPATH=.` avant l'exécution
- Exemple : `$env:PYTHONPATH = "." ; python -m src.scripts.script_name`

### Données manquantes
- Vérifiez la présence du dataset baseline dans `../baseline-CIC_IOT_2023/`
- Assurez-vous que les artifacts sont dans `artifacts/`

### Erreurs de preprocessing
- Vérifiez que les partitions ont été créées (étape 1)
- Contrôlez les logs dans `outputs/logs/preprocessor.log`

### Problèmes Flower
- Vérifiez que les ports 8080/9092 ne sont pas utilisés
- Assurez-vous que tous les terminaux sont démarrés dans l'ordre
- Consultez les logs serveur/client pour les détails