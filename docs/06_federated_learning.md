# Federated Learning

QI-FL-IDS-IoT uses Federated Learning to keep raw IoT traffic data local.

Core idea:

1. Each client trains on local data.
2. Clients send model updates to the FL server.
3. The server aggregates updates.
4. The global model is redistributed.

The final practical model uses FedAvg with QGA-selected features.

Important boundaries:

- Raw traffic data remains local.
- Secure aggregation and encrypted model updates are not implemented.
- The framework is a research implementation, not a production privacy guarantee.

Flower is the main FL runtime used by the project.
