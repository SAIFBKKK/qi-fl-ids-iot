# Scenario Simulator

Dry-run placeholder for controlled feature replay scenarios.

The simulator does not generate network traffic in P16.1 step 0. It only prints
the MQTT topic and JSON payload that later steps may publish inside the isolated
lab.

Example:

```bash
python scenario_publisher.py --dry-run --node-id iot-rpi-weak --scenario benign
```

