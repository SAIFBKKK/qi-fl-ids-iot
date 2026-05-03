# Plan de démo jury - QI-FL-IDS-IoT

Durée totale : 8-10 minutes

## Acte 1 - Architecture (1 min)

- Ouvrir http://localhost:8090
- Pointer la status pill verte "4/4 services"
- Pointer les 4 onglets
- Mentionner : "tout le système tourne en 9 microservices Docker"

## Acte 2 - Onglet 1 Réseau IoT (2 min)

- Tableau 3 nœuds tier-aware (weak / medium / powerful)
- Insister sur les 3 MD5 distincts : preuve cryptographique
- Cliquer "+ Connecter un nœud" pour faire apparaître un 4e nœud avec tier auto-assigné
- Filtrer par tier pour démontrer la souplesse opérationnelle

## Acte 3 - Onglet 2 Federated Learning (2 min)

- 4 KPI cards live
- Section Planification YAML : montrer le scheduler
- Cliquer "Lancer training mock" pour mettre à jour `last_run_at`
- Mentionner : "MLflow tracke chaque round avec ses métriques"

## Acte 4 - Onglet 3 QI vs Classique (2 min)

- Tableau 8 métriques x 6 méthodes
- Cellules "best" en vert, baseline en gris
- Pills measured (baseline) vs expected (QI)
- Radar chart à 6 polygones
- Insister sur l'honnêteté scientifique : "aucune valeur QI inventée, toutes citées des papiers (Barati 2024, QI 2025, Bhatia 2025)"

## Acte 5 - Monitoring + plan B (1 min)

- Onglet 4 Grafana embed avec dashboards live
- Mentionner Prometheus et MLflow accessibles directement
- Si Docker plante : sélecteur de scénario en haut, mode replay

## Plan B - vidéo

- Capturer une session complète à J-3 en condition stable
- Backup sur clé USB
- Si Docker plante en soutenance : "je vous montre la version enregistrée"
