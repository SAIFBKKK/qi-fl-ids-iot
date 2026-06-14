# P16 Live IoT Node Agent

Agent intended to run inside a VirtualBox IoT VM. It collects local hardware
metadata, registers with the live lab controller, publishes controlled feature
vectors to MQTT, and prints received alerts from `ids/alerts/{node_id}`.

Target command:

```bash
python agent.py --node-id iot-drone-sitl --server-url http://SERVER_IP:8020 --mqtt-host SERVER_IP --input-mode original_28_scaled
```

Supported input modes:

- `selected_12_scaled`: publish the final 12 selected scaled features directly.
- `original_28_scaled`: publish 28 original scaled features; `final-ids-api`
  applies the QGA mask internally.

The agent does not generate network traffic. It only publishes JSON feature
payloads to the local MQTT broker configured for the isolated lab.


