# Plan de migration fl-server → FastAPI
**Date :** 2026-05-02 | À valider avant implémentation

---

## 1. Routes à porter (compat ascendante garantie)

| Méthode | Path | Source actuelle | Changement |
|---|---|---|---|
| GET | `/health` | `_RegistryHandler.do_GET` ligne 107 | Porté tel quel |
| GET | `/nodes` | ligne 119 | Porté tel quel |
| GET | `/models` | ligne 122 | **ÉTENDU** — lit Model Factory |
| GET | `/models/{tier}/metadata` | lignes 125–131 | **ÉTENDU** — lit model_config.json + md5 |
| POST | `/nodes/register` | `_RegistryHandler.do_POST` ligne 134 | Porté tel quel |

---

## 2. Logique du bloc `main()` / module-level → lifespan

| Élément | Emplacement actuel | Emplacement FastAPI |
|---|---|---|
| `configure_logging()` | `main()` | Avant création `app`, au niveau module |
| `_fl_metrics = FLServerMetrics()` | Module-level singleton | Module-level singleton (inchangé) |
| `_node_registry = NodeRegistry()` | Module-level singleton | Module-level singleton (inchangé) |
| `_model_registry = ModelRegistry()` | Module-level singleton | **Remplacé** par `FactoryModelRegistry` |
| `_start_metrics_server()` | Appelé dans `run_registry_server()` | Appelé dans `lifespan` startup |
| `_update_registry_metrics()` | Appelé dans `run_registry_server()` | Appelé dans `lifespan` startup |
| `TRAINING_MODE` dispatch | `main()` switch | Dans `lifespan` : si mode ≠ registry, lance thread pour mock/real |
| `KEEP_SERVER_ALIVE` loop | `keep_alive_after_training()` | Dans thread de mock/real training |

**Règle R7 :** mock training et real training conservés intacts — lancés dans des threads daemon depuis le lifespan si `TRAINING_MODE` != registry.

---

## 3. Séparation des deux serveurs

```
Port 8080 — FastAPI (uvicorn)
    routes HTTP (/health, /nodes, /models, /nodes/register)
    lancé par CMD: uvicorn server_entrypoint:app --host 0.0.0.0 --port 8080

Port 8000 — Prometheus metrics (thread daemon)
    _start_metrics_server() — HTTPServer stdlib inchangé
    scrape target pour prometheus.yml : fl-server:8000/metrics
```

Les deux ports sont isolés. Port 8000 reste le serveur threading existant (R8 : pas de fusion).

---

## 4. Extension GET /models — lecture dynamique Model Factory

```python
MODEL_FACTORY_PATH = Path(os.getenv("MODEL_FACTORY_PATH", "/artifacts/model_factory_30rounds"))
TIERS = ["weak", "medium", "powerful"]

def list_available_tiers() -> list[dict]:
    if not MODEL_FACTORY_PATH.exists():
        return []
    tiers = []
    for tier_name in TIERS:
        tier_dir = MODEL_FACTORY_PATH / tier_name
        config_path = tier_dir / "model_config.json"
        model_path = tier_dir / "global_model.pth"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        tiers.append({
            "tier": tier_name,
            "available": True,
            "model_path": str(model_path),
            "scaler_path": str(tier_dir / "scaler.pkl"),
            "feature_names_path": str(tier_dir / "feature_names.json"),
            "label_mapping_path": str(tier_dir / "label_mapping.json"),
            "config": config,
            "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            "md5": hashlib.md5(model_path.read_bytes()).hexdigest()[:8] if model_path.exists() else None,
        })
    return tiers
```

Réponse GET /models :
```json
{
  "tiers": [...],
  "factory_path": "/artifacts/model_factory_30rounds",
  "factory_available": true
}
```

Si `MODEL_FACTORY_PATH` absent : `factory_available: false`, `tiers: []` — pas de 500.

---

## 5. Extension GET /models/{tier}/metadata

Retourne : config JSON + size_bytes + md5 + label_mapping résumé (nombre de classes).

---

## 6. Pydantic models nouveaux

- `TierInfo` : tier, available, model_path, size_bytes, md5, config
- `ModelsList` : tiers: list[TierInfo], factory_path, factory_available
- `TierMetadata` : TierInfo + label_mapping_summary (nb classes, classes list)
- `NodeRegistrationRequest` : node_id, cpu_cores, ram_mb, device_type, network_quality, battery_powered, tier_override
- `NodeRegistrationResponse` : node_id, assigned_tier, model_version, model_source, status

---

## 7. CORS

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

---

## 8. Dockerfile

Ajouts à `requirements.txt` :
- `fastapi>=0.111`
- `uvicorn[standard]>=0.29`
- `pydantic>=2.0` (probablement déjà présent via mlflow)

CMD → `["uvicorn", "server_entrypoint:app", "--host", "0.0.0.0", "--port", "8080"]`

---

## 9. docker-compose.yml — section fl-server

Ajouts :
- Port `8000:8000` exposé (metrics)
- Volume `:ro` pour model_factory_30rounds
- Env `MODEL_FACTORY_PATH=/artifacts/model_factory_30rounds`
- Healthcheck sur `/health`
