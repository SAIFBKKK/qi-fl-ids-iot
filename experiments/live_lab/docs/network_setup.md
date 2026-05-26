# Network Setup Plan

P16.1 only documents the future network.

Required server ports for the VMs:

- `1883`: MQTT broker.
- `8020`: live-lab-controller.
- `8014`: final-ids-api, optional direct readiness check.
- `8016`: final-mqtt-bridge readiness.
- `8015`: online-validator evidence endpoint.

MQTT topics:

- `ids/flows/{node_id}`
- `ids/predictions/{node_id}`
- `ids/alerts/{node_id}`
- `ids/status/{node_id}`

The network scope is local and isolated. No external targets are part of the
demo.

