# Audit de faisabilité — Dashboard QI-FL-IDS-IoT
**Date :** 2026-05-02 | **Auditeur :** Claude Sonnet 4.6 | **Branche :** feat/microservices

---

## PARTIE 1 — INVENTAIRE DES SERVICES

### 1. mosquitto
- **Chemin :** `services/mosquitto/`
- **Port :** `1883:1883`
- **Profile :** default (toujours actif)
- **Healthcheck :** ✅ (`mosquitto_sub` sur `$SYS/broker/version`)
- **HTTP Endpoints :** aucun
- **MQTT :** broker global ; topics gérés par les clients
- **Prometheus :** aucune
- **ENV :** `MQTT_USERNAME=ids_user`, `MQTT_PASSWORD`
- **État :** ✅ implémenté

### 2. iot-node
- **Chemin :** `services/iot-node/`
- **Port :** `8001:8000` (node-1), `8002:8000` (node-2, profil multi-node), `8003:8000` (node-3, profil multi-node)
- **Profile :** default (node-1) / `multi-node` (node-2, node-3)
- **Healthcheck :** ✅ (`GET /health`)
- **HTTP Endpoints :** `GET /health`, `GET /ready`, `GET /metrics`
- **MQTT Publish :** `ids/flows/{node_id}`, `ids/predictions/{node_id}`, `ids/alerts/{node_id}`, `ids/status/{node_id}`
- **MQTT Subscribe :** `ids/flows/{node_id}`
- **Prometheus :** Histogram `inference_latency_seconds` ; Counters `ids_flows_received_total`, `ids_predictions_total`, `ids_alerts_total` ; Gauges `ids_node_status`, `ids_node_assigned_tier_info`
- **ENV clés :** `NODE_ID`, `DEVICE_TYPE`, `NETWORK_QUALITY`, `BATTERY_POWERED`, `NODE_TIER_OVERRIDE`, `MODEL_PATH`, `SCALER_PATH`, `INFERENCE_THRESHOLD`
- **État :** ✅ implémenté (FastAPI + MQTT + inférence complète)

### 3. traffic-generator
- **Chemin :** `services/traffic-generator/`
- **Port :** `8010:8000`
- **Profile :** default
- **Healthcheck :** ✅ (`GET /health`)
- **HTTP Endpoints :** `GET /health`
- **MQTT Publish :** `ids/flows/{node_id}`, `ids/status/{node_id}`
- **Prometheus :** Counter `traffic_generator_flows_published_total`, `traffic_generator_rows_skipped_total` ; Gauge `traffic_generator_mqtt_connected`
- **ENV clés :** `NODE_ID`, `REPLAY_SCENARIO`, `REPLAY_RATE`, `DATASET_DIR`
- **État :** ✅ implémenté (replay Parquet)

### 4. fl-server
- **Chemin :** `services/fl-server/`
- **Port :** `8080:8080` (API) + `8000:8000` (metrics Prometheus interne)
- **Profile :** default (restart: "no")
- **Healthcheck :** ❌ absent
- **HTTP Endpoints :**
  - `GET /health`
  - `GET /nodes` — liste des nœuds enregistrés avec tiers
  - `GET /models` — liste des modèles par tier
  - `GET /models/{tier}/metadata`
  - `POST /nodes/register` — enregistrement IoT node
  - `GET /metrics` (port 8000, format Prometheus)
- **MQTT :** aucun
- **Prometheus :** Gauges `fl_current_round`, `fl_round_accuracy`, `fl_benign_recall`, `fl_f1_macro`, `fl_round_duration_seconds`, `fl_active_clients`, `registered_nodes_total`, `registered_nodes_by_tier`
- **ENV clés :** `TRAINING_MODE` (registry/mock/real), `FL_NUM_ROUNDS`, `MLFLOW_TRACKING_URI`, `REAL_FL_EXPERIMENT`, `KEEP_SERVER_ALIVE`
- **État :** ✅ implémenté

### 5. fl-client
- **Chemin :** `services/fl-client/`
- **Port :** aucun exposé
- **Profile :** `training` (3 réplicas : fl-client-1/2/3)
- **Healthcheck :** ❌ absent
- **HTTP Endpoints :** aucun (Flower gRPC)
- **MQTT :** aucun
- **Prometheus :** aucune
- **ENV clés :** `CLIENT_ID`, `FL_SERVER_ADDRESS`, `TRAINING_MODE` (mock/real), `MOCK_NUM_EXAMPLES`
- **État :** ✅ implémenté (mock Flower client)

### 6. qga-service
- **Chemin :** `services/qga-service/`
- **Port :** `8020:8000`
- **Profile :** `preprocessing`
- **Healthcheck :** ✅ (`GET /health`)
- **HTTP Endpoints :** `GET /health`, `GET /ready`, `POST /optimize`, `GET /metrics`
- **MQTT :** aucun
- **Prometheus :** Histogram `qga_optimization_latency_seconds` ; Counter `qga_requests_total` ; Gauges `qga_last_score`, `qga_service_status`
- **ENV clés :** `QGA_DEFAULT_ITERATIONS`
- **État :** 🔄 **STUB** — `services/qga-service/optimizer.py` lignes 30–35 : mode `"deterministic_stub"` explicite, aucune logique quantique/génétique réelle

### 7. edge-ids-gateway
- **Chemin :** `services/edge-ids-gateway/`
- **Port :** `8030:8000`
- **Profile :** `gateway`
- **Healthcheck :** ✅ (`GET /ready`, start_period=20s)
- **HTTP Endpoints :** `GET /`, `GET /health`, `GET /ready`, `GET /diagnostics`, `POST /validate/raw`, `POST /map/features`, `POST /infer/raw`, `GET /metrics`
- **MQTT Publish :** `iot/accepted/{gw_id}`, `iot/blocked/{gw_id}`, `ids/predictions/{gw_id}`, `ids/alerts/{gw_id}`, `ids/status/gateway/{gw_id}`
- **MQTT Subscribe :** `iot/raw/{gw_id}`
- **Prometheus :** Counters `edge_gateway_requests_total`, `edge_gateway_allowed_total`, `edge_gateway_blocked_total`, `edge_gateway_alerts_total` ; Histogram `edge_gateway_inference_latency_seconds` ; Gauges `edge_gateway_status`, `edge_gateway_ready`
- **ENV clés :** `GATEWAY_ID`, `INFERENCE_THRESHOLD`, `MODEL_PATH`, `SCALER_PATH`, `FEATURE_NAMES_PATH`
- **État :** ✅ implémenté

### 8. feature-extractor
- **Chemin :** `services/feature-extractor/`
- **Port :** `8000` interne — **aucun mapping host** dans docker-compose
- **Profile :** ❌ **absent des profils** (service non inclus dans docker-compose principal)
- **Healthcheck :** ✅ (défini dans Dockerfile)
- **HTTP Endpoints :** `GET /health`, `GET /metrics`
- **MQTT Publish :** `iot/features`
- **MQTT Subscribe :** `iot/raw_events` (fenêtre 60s)
- **Prometheus :** serveur HTTP manuel, pas de `prometheus_client`
- **ENV :** `NODE_ID`
- **État :** ✅ implémenté (28 features, 54 tests) mais **non branché** au compose

### 9. node-red
- **Chemin :** `services/node-red/`
- **Port :** `1880:1880`
- **Profile :** `orchestration`
- **Healthcheck :** ❌ absent
- **HTTP Endpoints :** dashboard Node-RED (port 1880)
- **MQTT Publish :** `iot/raw/node1` (injection de `raw_iot_event` toutes les 10s)
- **Prometheus :** aucune
- **ENV :** aucune
- **État :** ✅ implémenté (`services/node-red/flows.json`)

### 10. monitoring (Prometheus + Grafana)
- **Chemin :** `services/monitoring/`
- **Ports :** `9090:9090` (Prometheus), `3000:3000` (Grafana)
- **Profile :** default
- **Healthcheck :** ❌ absent (images standard)
- **Scrape targets Prometheus** (`services/monitoring/prometheus.yml`) :
  - `iot-node-1:8000`, `iot-node-2:8000`, `iot-node-3:8000`
  - `traffic-generator:8000`, `edge-ids-gateway:8000`
  - `feature-extractor:8000` ← **cible orpheline** (service non dans compose)
  - `fl-server:8000`
- **Prometheus :** scrape interval 15s, query API REST exposée
- **Grafana :** admin/qi-lab-2026, datasource Prometheus, dashboards provisionnés dans `services/monitoring/grafana/dashboards/`
- **État :** ✅ implémenté

---

## PARTIE 2 — MATRICE BESOINS DASHBOARD vs SERVICES

### Onglet 1 — Réseau IoT

| Besoin dashboard | Source idéale | Existe ? | Action requise |
|---|---|---|---|
| B1.1 Liste des nœuds enregistrés (5 max) | `GET /nodes` sur fl-server:8080 | ✅ | Aucune — route existe |
| B1.2 Profil hardware (cpu_cores, ram_mb, device_type) | Payload stocké lors du `POST /nodes/register` | 🔄 | Vérifier que fl-server persiste le profil complet dans sa registry mémoire et l'expose via `GET /nodes` (à confirmer dans `services/fl-server/server_entrypoint.py`) |
| B1.3 Tier assigné (weak/medium/powerful) | `GET /nodes` → champ `assigned_tier` | ✅ | Aucune |
| B1.4 Modèle déployé (chemin bundle, version) | `GET /models/{tier}/metadata` | ✅ | Aucune |
| B1.5 État de connexion (connected/disconnected) | Gauge `ids_node_status` via Prometheus | 🔄 | Ajouter logique heartbeat/timeout dans iot-node ou lire topic MQTT `ids/status/{node_id}` — l'état binaire existe mais pas le concept "disconnected" après timeout |
| B1.6 Bouton "Connecter" → POST /register | `POST /nodes/register` sur fl-server:8080 | ✅ | Ajouter CORS (voir T1) |
| B1.7 Métriques temps réel par nœud (latence, alertes/min) | Prometheus : `inference_latency_seconds`, `ids_alerts_total` | ✅ | Requête PromQL depuis dashboard ; Prometheus exposé sur :9090 |
| B1.8 3-5 nœuds simultanés | Profil `multi-node` (node-2, node-3) | 🔄 | Scaler à 5 réplicas dans compose ou activer profil `multi-node` (2 nœuds additionnels déjà prêts) |

### Onglet 2 — Federated Learning

| Besoin dashboard | Source idéale | Existe ? | Action requise |
|---|---|---|---|
| B2.1 Liste des runs MLflow (id, date, status) | MLflow REST API : `GET /api/2.0/mlflow/runs/search` sur :5000 | ✅ | Aucune — MLflow exposé sur :5000 |
| B2.2 Métriques par round (f1_macro, accuracy, benign_recall, loss) | MLflow run metrics ou Prometheus Gauges `fl_round_accuracy`, `fl_f1_macro`, `fl_benign_recall` | ✅ | Aucune — double source disponible |
| B2.3 Nombre de clients participants | Gauge Prometheus `fl_active_clients` | ✅ | Aucune |
| B2.4 Round courant (live) si training en cours | Gauge Prometheus `fl_current_round` | ✅ | Aucune |
| B2.5 Historique des trainings (timeline) | MLflow experiments list (`/api/2.0/mlflow/experiments/list`) | ✅ | Aucune |
| B2.6 Schedule du prochain training (cron) | Aucune source — ni cron, ni config exposée | ❌ | Créer `services/fl-server/schedule.yaml` avec config prochaine session + ajouter `GET /schedule` dans `services/fl-server/server_entrypoint.py` (0.5h + 1h) |

### Onglet 3 — QI vs Classique

| Besoin dashboard | Source idéale | Existe ? | Action requise |
|---|---|---|---|
| B3.1 Métriques baseline FedAvg (sans QI) | MLflow runs + artefacts dans `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/` | 🔄 | Données partielles : `.pkl` exist, logs MLflow existent — créer `experiments/fl-iot-ids-v3/baselines.yaml` avec valeurs consolidées (1h) |
| B3.2 Métriques QGA (M2) | qga-service stub → **aucune métrique réelle** | ❌ | Implémenter QGA réel dans `services/qga-service/optimizer.py` (4h minimum) **OU** utiliser valeurs simulées YAML en attendant |
| B3.3 Métriques QIARM (M1) | INCONNU — à confirmer avec Saif. Aucun service `qiarm` ou fichier nommé QIARM trouvé dans le repo | ❌ | Créer service ou YAML de valeurs selon papier de référence |
| B3.4 Métriques FedTN (M3) | INCONNU — à confirmer avec Saif. Aucun service `fedtn` trouvé | ❌ | Idem QIARM |
| B3.5 Source valeurs "expected" des papiers | `experiments/fl-iot-ids-v3/configs/global.yaml` — **pas de métriques papier** | ❌ | Créer `experiments/fl-iot-ids-v3/paper_baselines.yaml` avec valeurs annotées (1h) |

### Transversal

| Besoin | Source | Existe ? | Action requise |
|---|---|---|---|
| T1 CORS sur fl-server | `services/fl-server/server_entrypoint.py` | ❌ | Ajouter `CORSMiddleware` ou headers CORS dans HTTP handler — migrer vers FastAPI recommandé (2h) ou patch manuel `Access-Control-Allow-Origin` (0.5h) |
| T2 feature_names.json | Glob `**/*feature_names*` | ❌ | Seuls des `.pkl` existent (`experiments/fl-iot-ids-v3/outputs/.../feature_names.pkl`). Ajouter script de conversion pkl→json ou écrire directement (0.5h) |
| T3 Authentification API | Grep `auth|token|API_KEY` sur fl-server | ❌ | Aucune auth. Hors scope MVP dashboard mais risque sécurité (voir Partie 5) |
| T4 MLflow API REST | `http://localhost:5000` (port 5000 exposé) | ✅ | Aucune — API REST MLflow v2 accessible |
| T5 Prometheus query API | `http://localhost:9090/api/v1/query` | ✅ | Aucune — API exposée, pas d'auth |

---

## PARTIE 3 — VERDICT DE FAISABILITÉ

**Q1 : Le dashboard est-il réalisable avec les services actuels ?**
> **Oui avec ajouts mineurs** pour les onglets 1 et 2. L'onglet 3 (QI vs Classique) nécessite soit une implémentation QGA réelle, soit la création manuelle de fichiers YAML de référence — c'est un **refonte partielle** limitée à la couche données, pas à l'architecture.

**Q2 : Quel est le service le plus en retard ?**
> **qga-service** — stub déterministe explicite, aucune donnée réelle de comparaison QI disponible. C'est le bloquant principal de l'onglet 3.

**Q3 : Quelle est la donnée la plus difficile à obtenir pour l'onglet 3 ?**
> Les métriques **QIARM (M1)** et **FedTN (M3)** : aucun service, aucun fichier, aucune trace dans le repo. Si ces modèles ne sont pas encore entraînés, les valeurs doivent venir de la littérature (YAML manuel) ou d'expériences à lancer.

**Q4 : Risque de modifier l'architecture FL dans experiments/ ?**
> Non — tout peut rester côté `services/`. Les expériences dans `experiments/fl-iot-ids-v3/` n'ont pas besoin d'être modifiées ; seuls des fichiers YAML de consolidation sont à ajouter à leur côté.

---

## PARTIE 4 — PLAN D'ACTION CHIFFRÉ

| # | Tâche | Fichier(s) impactés | Estim. | Bloquant pour | Risque |
|---|---|---|---|---|---|
| 1 | Ajouter CORS headers sur fl-server (patch minimal) | `services/fl-server/server_entrypoint.py` | 0.5h | Onglets 1, 2 | Faible |
| 2 | Vérifier et exposer profil hardware complet dans `GET /nodes` | `services/fl-server/server_entrypoint.py` | 0.5h | Onglet 1 (B1.2) | Faible |
| 3 | Convertir `feature_names.pkl` → `feature_names.json` (script one-shot) | `experiments/fl-iot-ids-v3/outputs/…/feature_names.pkl` → nouveau `.json` | 0.5h | Transversal (T2) | Faible |
| 4 | Ajouter logique "disconnected" (timeout heartbeat) dans iot-node | `services/iot-node/` (endpoint `/health` ou Gauge MQTT) | 1h | Onglet 1 (B1.5) | Moyen |
| 5 | Activer profil `multi-node` ou scaler à 3-5 nœuds dans compose | `docker-compose.yml` | 0.5h | Onglet 1 (B1.8) | Faible |
| 6 | Ajouter `GET /schedule` + créer `schedule.yaml` sur fl-server | `services/fl-server/server_entrypoint.py`, nouveau `services/fl-server/schedule.yaml` | 1.5h | Onglet 2 (B2.6) | Faible |
| 7 | Créer `experiments/fl-iot-ids-v3/baselines.yaml` (consolidation métriques FedAvg) | Nouveau fichier YAML | 1h | Onglet 3 (B3.1) | Faible |
| 8 | Créer `experiments/fl-iot-ids-v3/paper_baselines.yaml` (valeurs QIARM, FedTN, QGA selon papiers) | Nouveau fichier YAML | 1h | Onglet 3 (B3.3–B3.5) | Moyen (dépend des papiers source) |
| 9 | Ajouter `GET /baselines` sur fl-server (lit le YAML ci-dessus) | `services/fl-server/server_entrypoint.py` | 1h | Onglet 3 | Faible |
| 10 | Implémenter QGA réel dans qga-service (si nécessaire pour onglet 3 live) | `services/qga-service/optimizer.py` | 4h | Onglet 3 (B3.2) | Élevé (logique métier complexe) |
| 11 | Ajouter feature-extractor au docker-compose principal (port 8040) | `docker-compose.yml` | 0.5h | Transversal | Faible |
| 12 | Ajouter healthcheck sur fl-server | `docker-compose.yml` | 0.5h | Stabilité | Faible |

**Total estimé :** 12h (sans tâche 10) à 16h (avec QGA)
**Conversion :** 1.5 à 2 jours-homme (base 8h/jour)

> Si l'onglet 3 utilise des valeurs statiques YAML (sans QGA live), le dashboard est livrable en **1.5 jours-homme**.
> Si QGA live est requis, compter **2.5 jours-homme** minimum.

---

## PARTIE 5 — RISQUES & POINTS D'ATTENTION

1. **Cibles Prometheus orphelines** — `services/monitoring/prometheus.yml` scrape `feature-extractor:8000` et `iot-node-2:8000`/`iot-node-3:8000`, mais ces services ne sont actifs que sous des profils spécifiques (`multi-node`, absent pour feature-extractor). En profil `default`, ces targets seront en erreur `connection refused` dans Prometheus — les dashboards Grafana associés afficheront "No data".

2. **fl-server sans CORS** — `services/fl-server/server_entrypoint.py` utilise `http.server.HTTPServer` (stdlib Python), pas FastAPI. Toute requête cross-origin depuis le dashboard React/Vue sera bloquée par le navigateur. Correction priorité 1.

3. **qga-service stub bloque l'onglet 3** — `services/qga-service/optimizer.py` retourne systématiquement `mode="deterministic_stub"`. Sans données QGA réelles ou YAML de référence, l'onglet comparatif QI vs Classique ne peut afficher que des placeholders. Risque de confusion utilisateur si non signalé dans l'UI.

4. **QIARM et FedTN introuvables** — Aucun service, fichier, ou run MLflow nommé `qiarm` ou `fedtn` identifié dans le repo. Si ces modèles sont des contributions originales du projet (et non des baselines issues de papiers), leur implémentation est **hors périmètre actuel** et devra être planifiée séparément.

5. **Worktree git non propre au moment de l'audit** — `git status` indique des fichiers modifiés non commités (`docs/reports/MODEL_FACTORY_30ROUNDS_REPORT.md`, `experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/run_status.json`) et des fichiers non trackés (`model_factory_summary.json`, répertoire `weak/`). Risque de perte si le développeur switche de branche sans commit. À nettoyer avant de démarrer les modifications du plan d'action.

---

*Rapport généré automatiquement — chemins vérifiés sur branche `feat/microservices` au 2026-05-02.*
