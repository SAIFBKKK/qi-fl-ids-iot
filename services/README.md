# Microservices stack - QI-FL-IDS-IoT

Architecture microservices pour le framework QI-FL-IDS-IoT.
9 services always-on + 2 services profiles.

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
Node-RED reste reserve au profile `orchestration`. Le FL server et QGA sont des
profiles futurs, non implementes en P4.

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

## Services

| Service | Port | Profile | Role |
|---|---:|---|---|
| traffic-generator | 8010 | default | Replay CIC-IoT-2023 |
| iot-node-1/2/3 | 8001/2/3 | default | Inference + FL client |
| fl-server | 8080 | training | FL orchestrator |
| mosquitto | 1883 | default | MQTT broker |
| mlflow | 5000 | default | Experiment tracking |
| prometheus | 9090 | default | Metrics scraping |
| grafana | 3000 | default | Dashboards |
| node-red | 1880 | orchestration | Scenario orchestration |
| qga-service | - | preprocessing | QGA feature selection |

## Profiles

```bash
# Demo de base, Mode A
docker-compose --env-file .env up -d

# Demo avec scenarios Node-RED, Mode C
docker-compose --env-file .env --profile orchestration up -d

# Round FL training
docker-compose --env-file .env --profile training up fl-server

# QGA preprocessing
docker-compose --env-file .env --profile preprocessing run --rm qga-service
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

## Implementation Status

- [x] P1 - Foundation (squelette + scripts)
- [x] P2 - iot-node service
- [x] P3 - traffic-generator
- [x] P4 - Compose Mode A complet
- [ ] P5 - fl-server
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
