# iot-rpi-weak

Raspberry-like weak VM placeholder for P16.1.

Planned role:

- publish controlled 12 selected scaled features;
- rely on server-side inference through `final-mqtt-bridge` and `final-ids-api`;
- receive alerts on `ids/alerts/iot-rpi-weak`.

Step 0 command:

```bash
python run_node.py --dry-run
```

No network activity is started in this step.

