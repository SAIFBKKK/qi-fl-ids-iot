# feature-extractor

Service Python qui agrège des événements IoT bruts par fenêtre temporelle,
extrait un vecteur de 28 features CIC-IoT et le publie sur MQTT après scaling.

```
iot/raw_events  →  [WindowAggregator 60s]  →  [FeatureExtraction]
               →  [build_canonical_vector]  →  [StandardScaler]
               →  iot/features
```

---

## ⚠️ Hypothèses à vérifier avant déploiement

| # | Hypothèse | Comment vérifier |
|---|-----------|-----------------|
| **H1** | `feature_names.pkl` contient les mêmes 28 noms **dans le même ordre** que `CANONICAL_FEATURE_NAMES` dans `feature_map.py` | `python -c "import joblib; print(joblib.load('/artifacts/feature_names.pkl'))"` |
| **H2** | Le scaler est un **`StandardScaler`** (pas `RobustScaler`) — le code utilise `scaler.mean_` comme centre de remplissage | `python -c "import joblib; s=joblib.load('/artifacts/scaler.pkl'); print(type(s).__name__)"` → doit afficher `StandardScaler` |
| **H3** | 13 features canoniques ne sont **pas extractibles** depuis le schéma d'événement brut (flags TCP, Header_Length, Protocol Type, Duration/TTL) — elles sont remplies avec `scaler.mean_[i]` (→ 0 après scaling, neutre pour le modèle) | Acceptable pour un proxy ; enrichir le schéma d'entrée si ces features sont critiques |

---

## Schéma des messages

### Input — `iot/raw_events`

```json
{
  "timestamp":  1700000000.0,
  "src_ip":     "192.168.1.10",
  "dst_ip":     "10.0.0.1",
  "protocol":   "TCP",
  "src_port":   54321,
  "dst_port":   80,
  "pkt_count":  5,
  "byte_count": 1500,
  "direction":  "outbound"
}
```

Champs obligatoires : `timestamp`, `src_ip`, `dst_ip`.
Tous les autres ont des valeurs par défaut (protocol=TCP, ports=0, counts=1/0, direction=outbound).

### Output — `iot/features`

```json
{
  "schema_version": "1.0",
  "event_type":     "feature_vector",
  "node_id":        "extractor-1",
  "src_ip":         "192.168.1.10",
  "timestamp":      "2026-01-01T00:00:00Z",
  "window_size_s":  60,
  "event_count":    42,
  "feature_names":  ["flow_duration", "Header_Length", ...],
  "vector":         [-0.12, 0.34, ...]
}
```

`vector` contient 28 floats scalés (StandardScaler). Les features non extractibles
(13/28) valent 0 après scaling (remplies avec `scaler.mean_` avant transform).

---

## Mapping features calculées → canoniques

| Feature calculée | Feature canonique | Source dans l'événement brut |
|---|---|---|
| `flow_duration` | `flow_duration` | `last_ts - first_ts` |
| `rate` | `Rate` | `flow_count / window_size_s` |
| `iat_ns` | `IAT` | mean inter-arrival time (ns) |
| `sum_bytes` | `Tot sum` | Σ `byte_count` |
| `min_bytes` | `Min` | min(`byte_count`) |
| `std_bytes` | `Std` | σ(`byte_count`) |
| `flow_count` | `Number` | nombre d'événements |
| `http_ratio` | `HTTP` | fraction flows → port 80 |
| `https_ratio` | `HTTPS` | fraction flows → port 443 |
| `dns_ratio` | `DNS` | fraction flows → port 53 |
| `ssh_ratio` | `SSH` | fraction flows → port 22 |
| `tcp_ratio` | `TCP` | fraction flows protocol TCP |
| `udp_ratio` | `UDP` | fraction flows protocol UDP |
| `arp_ratio` | `ARP` | fraction flows protocol ARP |
| `icmp_ratio` | `ICMP` | fraction flows protocol ICMP |
| *(non mappable)* | `Header_Length`, `Protocol Type`, `Duration`, tous les flags/counts TCP | → `scaler.mean_[i]` |

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `NODE_ID` | `extractor-1` | Identifiant du service |
| `MQTT_BROKER` | `localhost` | Hostname du broker Mosquitto |
| `MQTT_PORT` | `1883` | Port MQTT |
| `MQTT_USERNAME` | *(vide)* | Authentification MQTT |
| `MQTT_PASSWORD` | *(vide)* | Authentification MQTT |
| `WINDOW_SIZE_S` | `60` | Taille de la fenêtre d'agrégation (secondes) |
| `SCALER_PATH` | `/artifacts/scaler.pkl` | Chemin vers le scaler (joblib) |
| `FEATURE_NAMES_PATH` | `/artifacts/feature_names.pkl` | Chemin vers la liste de features (`.pkl` ou `.json`) |
| `INPUT_TOPIC` | `iot/raw_events` | Topic MQTT d'entrée |
| `OUTPUT_TOPIC` | `iot/features` | Topic MQTT de sortie |
| `HEALTH_PORT` | `8000` | Port HTTP `/health` |
| `LOG_LEVEL` | `INFO` | Niveau de log (DEBUG/INFO/WARNING/ERROR) |

---

## Lancement local (sans Docker)

```powershell
$bundle = "C:\Users\saifb\dev\qi-fl-ids-iot\experiments\fl-iot-ids-v3\outputs\deployment\baseline_fedavg_normal_classweights"
$env:SCALER_PATH        = "$bundle\scaler.pkl"
$env:FEATURE_NAMES_PATH = "$bundle\feature_names.pkl"
$env:MQTT_BROKER        = "localhost"
$env:MQTT_PORT          = "1883"
$env:NODE_ID            = "extractor-1"
$env:WINDOW_SIZE_S      = "60"

cd services\feature-extractor
pip install -r requirements.txt
python extractor.py
```

Vérifier le health endpoint :
```powershell
Invoke-RestMethod http://localhost:8000/health
```

---

## Lancement avec Docker Compose (profile gateway)

```bash
cd services
docker compose --profile gateway up -d --build feature-extractor
```

Voir le snippet dans `docker-compose.yml` (service `feature-extractor`, profile `gateway`).

---

## Tests

```bash
cd services/feature-extractor
python -m pytest tests/ -v
```

33 tests unitaires, aucun broker réel requis.
Le test `test_real_bundle_scaler` est automatiquement sauté si le bundle n'est pas disponible.

---

## Changelog

### v2.0.0 — 2026-05-02
**Schéma raw_event v2 — couverture 28/28 features**

#### Changements majeurs
- Schéma raw_event enrichi : 7 nouveaux champs (fin_flag, syn_flag, rst_flag, psh_flag, ack_flag, urg_flag, header_length)
- feature_map.py v2 : couverture 28/28 features (vs 15/28 en v1)
  - Flags TCP : fin/syn/rst/psh/ack/urg (valeur + count agrégé par fenêtre)
  - Binaires protocole dérivés de dst_port : HTTP(80), HTTPS(443), DNS(53), SSH(22)
  - Binaires transport dérivés de protocol : TCP, UDP, ARP, ICMP
  - Protocol Type encodé entier (TCP=6, UDP=17, ICMP=1, ARP=0)
- Artefacts de référence gelés :
  - Scaler : experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/scaler.pkl (StandardScaler)
  - Features : experiments/fl-iot-ids-v3/artifacts/feature_names.pkl (28 features)
  - Modèle : global_model.pth input_dim=28
- Tests : 54/54 passent (vs 33/33 en v1)

#### Limitations documentées
- urg_count : fixé à 0 par défaut (quasi-zéro dans CIC-IoT-2023, justifié)
- Header_Length : défaut 20 si absent du raw_event
- Tous les champs v2 ont des defaults — rétrocompatible avec raw_event v1

### v1.0.0 — 2026-04-XX
- Version initiale : 15/28 features, 33 tests
