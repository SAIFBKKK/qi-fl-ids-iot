# Microservices stack - QI-FL-IDS-IoT

Architecture microservices pour le framework QI-FL-IDS-IoT.
Mode A fournit la demo IDS temps reel par defaut. Les modes avances restent
isoles via Docker Compose profiles.

## Mode A Demo

```bash
cd services

# 1. Preparer la configuration
cp .env.example .env
# Editer .env, au minimum changer MQTT_PASSWORD et GRAFANA_ADMIN_PASSWORD

# 2. Generer le password MQTT
bash scripts/generate_mqtt_password.sh

# 3. Construire les subsets de demo
../.venv/Scripts/python.exe scripts/build_demo_subsets.py

# 4. Demarrer la demo IDS temps reel
docker compose up -d --build

# 5. Verifier
bash scripts/demo_check.sh
```

Sur Windows PowerShell, la verification peut aussi se lancer avec :

```powershell
.\scripts\demo_check.ps1
```

Mode A lance la demo IDS temps reel sans FL training :
`traffic-generator -> MQTT -> iot-node-1 -> predictions/alerts -> Prometheus/Grafana`.
MLflow est disponible sur le port `5000`, mais reste passif dans Mode A.

Le scenario par defaut est `mixed_chaos` avec `REPLAY_RATE=5`.
Node-RED reste reserve au profile `orchestration`. Le training FL est separe
dans le Mode B, et QGA reste un profile futur.

Endpoints utiles :

- iot-node health: <http://localhost:8001/health>
- traffic-generator health: <http://localhost:8010/health>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>
- MLflow: <http://localhost:5000>

Debug MQTT :

```bash
docker exec -it mosquitto mosquitto_sub \
  -h localhost -p 1883 \
  -u ids_user -P "$MQTT_PASSWORD" \
  -t "ids/#"
```

Arret :

```bash
docker compose down
```

## P7.8 - Edge IDS Gateway profile

Le profile `gateway` ajoute une gateway IDS edge optionnelle sans modifier le
Mode A par defaut. Mode A continue de lancer uniquement la chaine existante:

`traffic-generator -> MQTT -> iot-node-1 -> predictions/alerts`

Le profile gateway lance en plus `edge-ids-gateway`, qui recoit des evenements
IoT bruts sur MQTT, applique la validation raw, le mapping 28 features,
`scaler.pkl`, `global_model.pth`, puis publie allow/block/predictions/alerts.

Lancement:

```bash
cd services
docker compose --profile gateway up -d --build edge-ids-gateway
```

Lancement explicite avec le broker:

```bash
cd services
docker compose --profile gateway up -d --build mosquitto edge-ids-gateway
```

Endpoints gateway:

- health: <http://localhost:8030/health>
- readiness: <http://localhost:8030/ready>
- diagnostics: <http://localhost:8030/diagnostics>
- metrics: <http://localhost:8030/metrics>

Topics MQTT utilises:

- input brut: `iot/raw/node1`
- accepted: `iot/accepted/node1`
- blocked: `iot/blocked/node1`
- predictions: `ids/predictions/node1`
- alerts: `ids/alerts/node1`
- status gateway: `ids/status/gateway/node1`

Test Mosquitto:

```bash
docker exec -it mosquitto mosquitto_sub \
  -h localhost -p 1883 \
  -u ids_user -P "$MQTT_PASSWORD" \
  -t "ids/#"
```

Publier un evenement raw valide:

```bash
docker exec mosquitto mosquitto_pub \
  -h localhost -p 1883 \
  -u ids_user -P "$MQTT_PASSWORD" \
  -t iot/raw/node1 \
  -m '{"schema_version":"1.0","event_type":"raw_iot_event","timestamp":"2026-04-28T12:00:00Z","node_id":"sensor-a1","gateway_id":"node1","node_group":"room-a","device_type":"thermostat","src_ip":"10.10.1.21","dst_ip":"10.10.0.10","src_port":51544,"dst_port":443,"protocol":"tcp","packet_size":820,"packet_count":6,"duration_ms":85,"bytes_in":920,"bytes_out":3980,"flags":{"syn":1},"flag_counts":{},"scenario":"normal_traffic"}'
```

Arret:

```bash
docker compose --profile gateway down
```

Prometheus contient un job `edge-ids-gateway`. Il est normal que ce target soit
`DOWN` quand le profile `gateway` n'est pas lance.

Limites P7.8:

- Node-RED n'est pas encore branche
- le mapping raw -> 28 features reste une approximation deterministe documentee
- Mode A reste inchange

Nettoyage optionnel et destructif pour Docker local :

```bash
docker system prune -a --volumes -f
```

> **Note sur les chemins de donnees :**
> Le dataset complet CIC-IoT-2023 est dans
> `data/balancing_v3_fixed300k_outputs/` (gitignored).
> Les subsets de demo generes par `build_demo_subsets.py` sont dans
> `data/cic-iot-2023/demo_subsets/` (versionnes Git, 4.7 MB generes en P1).
> Le bundle FL baseline est dans
> `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/`
> (versionne Git, environ 200 KB).

## Mode B FL Training Profile

Mode B lance un profile Docker Compose separe avec deux modes :

- `TRAINING_MODE=mock` : orchestration Flower mock P5 avec 1 serveur + 3 clients.
- `TRAINING_MODE=real` : wrapper P6A-lite autour du vrai runner scientifique `run_experiment.py`.

```bash
cd services

# Mode mock par defaut : MLflow + serveur Flower mock + 3 clients Flower mock
docker compose --profile training up -d --build

# Verifier le profile training
bash scripts/training_check.sh
```

Mode real P6A-lite :

```bash
cd services

# Real scientific runner, limite a 1 round par defaut dans .env.example
TRAINING_MODE=real REAL_FL_ROUNDS=1 docker compose --profile training up -d --build

# Verifier le profile training
bash scripts/training_check.sh
```

### P5B - MLflow container path fix

The real FL runner is executed through a thin Docker-side wrapper that patches
`run_experiment.resolve_tracking_uri` at runtime. This avoids Windows host paths
such as `C:\...` leaking into the Linux container and being interpreted as
`/C:`. The scientific source code under `experiments/fl-iot-ids-v3/src` remains
unchanged.

Sur Windows PowerShell :

```powershell
.\scripts\training_check.ps1
```

Logs utiles :

```bash
docker logs fl-server
docker logs fl-client-1
docker logs fl-client-2
docker logs fl-client-3
```

Arret du Mode B :

```bash
docker compose --profile training down
```

Clarification importante :

- Mode A = inference IDS temps reel avec MQTT, modele PyTorch et metrics Prometheus.
- Mode B mock = profile d'orchestration FL avec Flower mock leger.
- Mode B real = wrapper scientifique simulation-based via `experiments/fl-iot-ids-v3/src/scripts/run_experiment.py`.
- P5 valide Docker/profile/training orchestration, pas les metriques scientifiques FL.
- P6A-lite branche le vrai runner Multi-tier valide mais ne demarre pas de vrais clients multi-containers.
- Le vrai Multi-tier FL valide reste dans `experiments/fl-iot-ids-v3/`.
- L'integration complete du vrai Multi-tier FL distribue pourra etre traitee dans une phase ulterieure.

## Services

| Service | Port | Profile | Role |
|---|---:|---|---|
| traffic-generator | 8010 | default | Replay CIC-IoT-2023 |
| iot-node-1 | 8001 | default | Inference IDS MQTT |
| fl-server | 8080 | training | Mock Flower FL server |
| fl-client-1/2/3 | - | training | Mock Flower FL clients |
| mosquitto | 1883 | default | MQTT broker |
| mlflow | 5000 | default | Experiment tracking |
| prometheus | 9090 | default | Metrics scraping |
| grafana | 3000 | default | Dashboards |
| node-red | 1880 | orchestration | Scenario orchestration |
| qga-service | 8020 | preprocessing | Quantum-inspired optimization API stub |
| edge-ids-gateway | 8030 | gateway | Edge IDS raw MQTT gateway |

## Profiles

```bash
# Demo de base, Mode A
docker compose up -d

# Demo avec scenarios Node-RED, Mode C
docker compose --profile orchestration up -d

# Mode B mock FL training profile
docker compose --profile training up -d --build

# QGA preprocessing API
docker compose --profile preprocessing up -d --build qga-service

# Edge IDS Gateway profile
docker compose --profile gateway up -d --build edge-ids-gateway
```

## Mode C Quantum-Inspired Preprocessing Profile

`qga-service` exposes a lightweight deterministic optimization API for demo and
integration purposes. It does not implement the full scientific QGA algorithm in
P6C, and it does not depend on the dataset.

```bash
cd services
docker compose --profile preprocessing up -d --build qga-service
curl http://localhost:8020/health
curl -X POST http://localhost:8020/optimize \
  -H "Content-Type: application/json" \
  -d '{"available_features":28,"latency_budget_ms":5.0,"energy_budget":0.75,"risk_tolerance":0.4}'
curl http://localhost:8020/metrics
```

## US8/US9 Validation

US8 and US9 are closed with the current microservices decomposition:

- `iot-node`: node-side IDS service with internal `collector`, `preprocessor`,
  `inference`, `metrics`, and HTTP health/readiness modules.
- `traffic-generator`: replay service that publishes CIC-IoT-2023 demo flows to
  MQTT.
- `fl-server` and `fl-client`: optional `training` profile for mock Flower
  orchestration and the real simulation-based scientific runner.
- `qga-service`: optional `preprocessing` profile exposing a deterministic
  quantum-inspired optimization API stub.

HTTP service endpoints:

| Service | Profile | Health | Readiness | Metrics |
|---|---|---|---|---|
| `iot-node-1` | default | `GET /health` | `GET /ready` | `GET /metrics` |
| `traffic-generator` | default | `GET /health` | `GET /ready` | `GET /metrics` |
| `qga-service` | preprocessing | `GET /health` | `GET /ready` | `GET /metrics` |

Validation commands:

```powershell
cd services
docker compose up -d --build
.\scripts\demo_check.ps1

docker compose --profile training up -d --build
.\scripts\training_check.ps1

docker compose --profile preprocessing up -d --build qga-service
Invoke-RestMethod http://localhost:8020/health
Invoke-RestMethod http://localhost:8020/ready
Invoke-RestMethod http://localhost:8020/metrics
```

On Bash-compatible shells:

```bash
cd services
docker compose up -d --build
bash scripts/demo_check.sh
docker compose --profile training up -d --build
bash scripts/training_check.sh
docker compose --profile preprocessing up -d --build qga-service
curl http://localhost:8020/health
curl http://localhost:8020/ready
curl http://localhost:8020/metrics
```

## Structure

Voir `services/<service>/README.md` pour chaque service individuel.

## Scripts Utilitaires

- `scripts/build_demo_subsets.py` - Genere les 6 fichiers parquet de demo
- `scripts/generate_mqtt_password.sh` - Cree le password file Mosquitto
- `scripts/reset.sh` - Reset propre soft ou complet avec `--hard`
- `scripts/healthcheck_all.sh` - Verifie l'etat des services infra P1
- `scripts/test_publish.py` - Publie des flows de test MQTT depuis les demo subsets
- `scripts/demo_check.sh` - Verifie la demo Mode A sous Bash
- `scripts/demo_check.ps1` - Verifie la demo Mode A sous PowerShell
- `scripts/training_check.sh` - Verifie le profile training Mode B sous Bash
- `scripts/training_check.ps1` - Verifie le profile training Mode B sous PowerShell

## Implementation Status

- [x] P1 - Foundation (squelette + scripts)
- [x] P2 - iot-node service
- [x] P3 - traffic-generator
- [x] P4 - Compose Mode A complet
- [x] P5 - Mock FL training profile
- [ ] P6 - Monitoring dashboards Grafana + qga-service
- [ ] P7 - Node-RED scenarios
- [ ] P8 - Tests E2E
- [ ] P9 - Documentation finale

## P2 Validation Snapshot

P2 a ete valide sur `iot-node-1` avec le bundle US1 reel et MQTT authentifie.

- Health: `/health` retourne `status=ok`, `mqtt_connected=true`, `inference_engine=torch_mlp`
- Pipeline MQTT: 5 flows `ddos_burst` recus, 5 predictions publiees, 5 alerts publiees
- Schema strict: 0 rejet, aucune imputation silencieuse
- Labels observes: `DDoS-UDP_Flood`, `DDoS-RSTFINFlood`, `DDoS-SlowLoris`, `DDoS-SYN_Flood`
- Latence CPU edge: environ 2 ms par flow, soit moins de 5 ms par inference
- Metrics Prometheus: `ids_flows_received_total`, `ids_flows_rejected_invalid_schema_total`, `ids_predictions_total`, `ids_alerts_total`, `inference_latency_seconds`, `ids_node_status`

## P3 Validation Snapshot

P3 a ete valide avec la chaine complete `traffic-generator -> MQTT -> iot-node-1 -> modele PyTorch -> predictions/alerts -> Prometheus`.

- Flux recus par `iot-node-1`: `ids_flows_received_total=16778`
- Predictions: `ids_predictions_total` augmente sur plusieurs classes CIC-IoT-2023
- Alerts: `ids_alerts_total` actif avec severites `low`, `medium`, `high`, `critical`
- Rejets schema/features: `0` sur toutes les raisons exposees
- Latence inference: `7.626942627s` pour `16778` flows, soit environ `0.45 ms/flow`
- Node status: `ids_node_status{node_id="node1"} 1`

## P4 Validation Snapshot

P4 a ete valide comme Mode A complet de demo IDS en une commande.

- Commande de lancement: `docker compose up -d --build`
- Verification: `bash scripts/demo_check.sh`
- Resultat final: `Mode A demo check: PASS`
- Services default valides: `mosquitto`, `iot-node-1`, `traffic-generator`, `prometheus`, `grafana`, `mlflow`
- Pipeline valide: `traffic-generator -> MQTT -> iot-node-1 -> predictions/alerts -> Prometheus/Grafana`
- MLflow: disponible sur <http://localhost:5000>, passif dans Mode A
- Hors scope P4: pas de FL training, pas de Node-RED, pas de QGA, pas de node2/node3
- Tag stable: `p4-mode-a-demo`

## P5 Validation Snapshot

Mode B - FL Training Profile valide.

Validation:

- Commande de lancement: `docker compose --profile training up -d --build`
- Verification: `bash scripts/training_check.sh`
- Resultat final: `training_check.sh -> PASS`

Services valides:

- `fl-server`
- `fl-client-1`
- `fl-client-2`
- `fl-client-3`
- `mlflow`

Observations:

- Flower server a complete 10 rounds
- 3/3 clients ont participe avec succes
- 0 failure runtime observee
- MLflow experiment cree: `p5_mock_fl_training`

Important:

Ce profile valide l'orchestration FL et la separation Docker Compose.
Il ne represente pas le benchmark scientifique FL.
Les vraies experiences scientifiques FL restent dans `experiments/fl-iot-ids-v3/`.

## P6A Validation Snapshot

Mode B real - runner scientifique Multi-tier execute dans le profile Docker.

Validation:

- Commande de lancement: `TRAINING_MODE=real REAL_FL_ROUNDS=10 docker compose --profile training up -d --build`
- Resultat final: `Scientific runner completed successfully`
- Conteneur `fl-server`: `Exited (0)`
- Duree Flower: `10 rounds in 384.07s`
- Duree wrapper Docker observee: environ `450.33s`

Observations systeme:

- Le bug MLflow `/C:` est corrige par le wrapper Docker-side.
- MLflow reste actif via `http://mlflow:5000`.
- Le runner scientifique reel est execute sans modifier `experiments/fl-iot-ids-v3/src/`.
- Les logs dupliques Flower/Ray de shutdown ne sont pas bloquants.

Resultats round 1 -> round 10:

| Metrique | Round 1 | Round 10 | Evolution |
|---|---:|---:|---:|
| Distributed loss | 1.2345 | 0.8903 | baisse nette |
| Train loss last | 0.8522 | 0.6902 | baisse nette |
| Accuracy | 0.5651 | 0.7155 | +0.1504 |
| Macro-F1 | 0.4862 | 0.6696 | +0.1834 |
| Recall macro | 0.5407 | 0.6885 | +0.1478 |
| Benign recall | 0.8254 | 0.8730 | +0.0476 |
| False positive rate | 0.1746 | 0.1270 | -0.0476 |
| Rare class recall | 0.1321 | 0.3306 | +0.1985 |
| Rare macro-F1 | 0.1588 | 0.2765 | +0.1177 |

Communication Multi-tier:

- Round 1 warm-up full-width: `536472` bytes
- Round 2+ reduced-width: `255768` bytes
- Reduction apres warm-up: environ `52.32%`
- Reduction totale estimee sur 10 rounds: environ `47.09%`

Point scientifique important:

Le round 10 ameliore les metriques globales, mais les classes rares culminent au
round 9 (`rare_class_recall=0.3767`, `rare_macro_f1=0.2976`) avant de redescendre
legerement au round 10. Ce comportement justifie une future selection du meilleur
round selon une metrique IDS orientee attaques rares, par exemple Rare Recall.

Interpretation:

P6A valide l'integration MLOps/Docker du runner reel. Les resultats scientifiques
de reference restent ceux de la meilleure campagne US6; le run P6A sert surtout
de preuve que le pipeline Multi-tier reel est executable dans l'environnement
microservices.
