# iot-drone-sitl compatibility path

This legacy directory name now points to the Phase 2 Live Lab drone SITL VM1 role.

Planned role:

- publish controlled 12 selected scaled MAVLink/UDP packet-window features;
- rely on server-side inference through `final-mqtt-bridge` and `final-ids-api`;
- receive alerts on `ids/alerts/iot-drone-sitl`.

Step 0 command:

```bash
python run_node.py --dry-run
```

No network activity is started in this step.


