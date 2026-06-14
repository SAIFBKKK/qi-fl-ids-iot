# Architecture

QI-FL-IDS-IoT separates scientific experimentation from deployment demonstration.

```text
IoT/WSN nodes
  -> local traffic preprocessing
  -> local model training or inference
  -> federated model update exchange
  -> global aggregation
  -> IDS API and monitoring
```

Main layers:

- **Data layer**: CICIoT2023-derived preprocessing, L1 binary IDS labels, and federated partitions.
- **Learning layer**: centralized baselines, FedAvg, QGA-selected FedAvg, QIFA research variants.
- **Quantum-inspired layer**: QGA, QIFA, and FedTN/MPS compression analysis.
- **Deployment layer**: Docker services, MQTT flow, feature extraction, IDS API, dashboard, Prometheus, Grafana.
- **Artifact layer**: generated outputs, models, scalers, logs, figures, and reports stored outside GitHub.

Public GitHub should contain source, configs, docs, tests, scripts, Docker files, selected small metadata, and tiny samples only.
