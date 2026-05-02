# Mode A Dynamic Model Assignment

## Ancien Mode A

Le Mode A actuel reste compatible:

```text
traffic-generator -> Mosquitto -> iot-node-1 -> predictions/alerts/status -> Prometheus/Grafana
```

`iot-node-1` charge localement le bundle bind-mounté dans `/artifacts`:

- `MODEL_PATH=/artifacts/global_model.pth`
- `SCALER_PATH=/artifacts/scaler.pkl`
- `LABEL_MAPPING_PATH=/artifacts/label_mapping.json`

L'inférence reste locale. Les messages MQTT `ids/predictions/{node_id}`, `ids/alerts/{node_id}` et `ids/status/{node_id}` ne changent pas.

## Nouveau Mode A dynamique

Au démarrage, chaque `iot-node` peut contacter `fl-server` si `MODEL_SERVER_URL` est défini.

```text
iot-node startup
  -> collect hardware profile
  -> POST fl-server /nodes/register
  -> receive assigned_tier: weak | medium | powerful
  -> expose assigned_tier in /health and /metrics
  -> continue local inference with /artifacts bundle
```

Si `MODEL_SERVER_URL` est absent ou indisponible, le node garde le fallback historique `/artifacts`.

## Endpoints fl-server

`fl-server` expose ces endpoints en `TRAINING_MODE=registry` sur `FL_SERVER_PORT`:

- `GET /health`
- `POST /nodes/register`
- `GET /nodes`
- `GET /models`
- `GET /models/{tier}/metadata`

Exemple registration:

```json
{
  "node_id": "iot-node-1",
  "cpu_cores": 2,
  "ram_mb": 1024,
  "device_type": "docker_node",
  "network_quality": "medium",
  "battery_powered": false,
  "tier_override": "weak"
}
```

Réponse:

```json
{
  "node_id": "iot-node-1",
  "assigned_tier": "weak",
  "model_version": "placeholder",
  "model_source": "local_registry",
  "status": "registered"
}
```

## Variables d'environnement

### fl-server

- `FL_SERVER_MODE=registry`: active le registry Mode A dynamique dans Compose.
- `TRAINING_MODE=registry`: valeur effective lue par le conteneur `fl-server`.
- `FL_SERVER_HOST=0.0.0.0`
- `FL_SERVER_PORT=8080`
- `LOG_LEVEL=INFO`
- `LOG_FORMAT=json`

`FL_SERVER_MODE=mock` et `FL_SERVER_MODE=real` restent disponibles pour les usages FL existants.

### iot-node

- `MODEL_SERVER_URL=http://fl-server:8080`: active registration dynamique.
- `NODE_ID=node1`
- `DEVICE_TYPE=docker_node`
- `NETWORK_QUALITY=medium`
- `BATTERY_POWERED=false`
- `NODE_TIER_OVERRIDE=weak`: override demo optionnel.
- `RAM_MB=1024`: fallback si `psutil` n'est pas disponible.

Les variables historiques `MODEL_PATH`, `SCALER_PATH`, `LABEL_MAPPING_PATH`, `MQTT_*` restent inchangées.

## Règles de tier

- `weak` si `cpu_cores <= 2` ou `ram_mb < 2048`
- `medium` si `cpu_cores <= 4` ou `ram_mb < 4096`
- `powerful` sinon
- `NODE_TIER_OVERRIDE` peut forcer `weak`, `medium` ou `powerful` pour la démo.

## Métriques Prometheus

### iot-node

- `ids_node_assigned_tier_info{node_id="node1",tier="weak"} 1`

### fl-server

- `registered_nodes_total`
- `registered_nodes_by_tier{tier="weak"}`
- `registered_nodes_by_tier{tier="medium"}`
- `registered_nodes_by_tier{tier="powerful"}`

## Validation manuelle

```powershell
docker compose -f services/docker-compose.yml config
docker compose -f services/docker-compose.yml up -d mosquitto fl-server iot-node-1 prometheus grafana

curl http://localhost:8080/health
curl http://localhost:8080/nodes
curl http://localhost:8001/health
curl http://localhost:8001/metrics
```

À vérifier:

- `fl-server` démarre en mode registry.
- `iot-node-1` s'enregistre.
- `assigned_tier` est visible dans `/health`.
- `ids_node_assigned_tier_info` est visible dans `/metrics`.
- Le chargement historique `/artifacts` fonctionne toujours.
- Les alertes MQTT et le scrape Prometheus `iot-node-1:8000` continuent de fonctionner.

## Limites actuelles

Le registry expose encore des métadonnées placeholder. Il n'impose pas encore le téléchargement dynamique d'un bundle.

## Prochaine étape

Brancher les vrais bundles `weak`, `medium`, `powerful` produits par `experiments/fl-iot-ids-v3` après la fin du training offline, puis ajouter la sélection/téléchargement effectif côté `iot-node`.
