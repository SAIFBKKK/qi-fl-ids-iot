# P16.9.2 Live Defense Demo Mode

## Goal

P16.9.2 adds a dedicated dashboard page for the final jury presentation:

- route: `http://192.168.56.1:8013/demo`
- state endpoint: `http://192.168.56.1:8013/api/live-lab/demo-state`
- mode name: `Live Defense Demo Mode`

The demo is safe by design. It uses VM registration and controlled PacketWindow(30) MQTT publication only. It does not run packet capture, lab attack scenarios, training, Flower, or any offensive command.

## Before The Defense

1. Start Docker Desktop on the server PC.
2. Start the live lab services from the server repository:

```bash
cd experiments/qi-fl-ids-iot-final/deployment
docker compose -f docker-compose.final.yml --profile online --profile live-lab up -d --build
```

3. Open the demo page:

```text
http://192.168.56.1:8013/demo
```

4. Check that the page shows the platform status and service readiness.
5. Check that VM1 and VM2 can reach the server IP `192.168.56.1`.

## Live Demonstration Flow

1. Show `Platform Ready`.
2. Register VM1:

```bash
python3 experiments/live_lab/nodes/iot_rpi_weak/run_node.py --server-url http://192.168.56.1:8020 --register
```

3. Register VM2:

```bash
python3 experiments/live_lab/nodes/iot_smart_watch_medium/run_node.py --server-url http://192.168.56.1:8020 --register
```

4. Explain the assigned tiers:

- `iot-drone-sitl` is assigned to tier `weak`.
- `iot-smart-watch-medium` is assigned to tier `medium`.

5. Explain the final model assignment:

- final model: `P8 FedAvg + QGA`
- model_id: `p8_fedavg_qga_l1`
- selected_mask_id: `conservative_seed_42`
- runtime scaler: JSON scaler packaged for the live lab VMs

6. Publish VM1 controlled PacketWindow(30) evidence:

```bash
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py --broker 192.168.56.1 --node-id iot-drone-sitl --input-mode selected_12_scaled --window-size 30 --max-windows 1 --publish
```

7. Show the VM1 prediction and alert in the dashboard.

8. Publish VM2 controlled PacketWindow(30) evidence:

```bash
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py --broker 192.168.56.1 --node-id iot-smart-watch-medium --input-mode original_28_scaled --window-size 30 --max-windows 1 --publish
```

9. Show the VM2 prediction and alert in the dashboard.
10. Show that runtime error counters remain at zero.
11. Open Grafana only if more infrastructure evidence is useful.

## Presentation Phrases

- "Here the weak IoT node is registered and assigned to the final model."
- "The smart-watch-like node sends 28 scaled features; the API applies the QGA mask."
- "The alert appears live after the MQTT bridge calls the final IDS model."
- "Runtime errors remain zero, which validates the deployment path."

## What The Jury Is Seeing

This live lab demonstrates that each IoT node is registered, assigned to a tier, linked to the final P8 FedAvg + QGA model, and able to send packet-window features through MQTT to obtain real IDS predictions and alerts.

The live dashboard separates scientific evaluation from runtime deployment evidence. Scientific metrics are from P12/P13, while this demo validates the operational path.

## Limits

- P16.9.2 does not change scientific results.
- P16.9.2 does not evaluate detection performance.
- P16.9.2 does not use live packet capture.
- P16.9.2 does not run lab attack scenarios.

