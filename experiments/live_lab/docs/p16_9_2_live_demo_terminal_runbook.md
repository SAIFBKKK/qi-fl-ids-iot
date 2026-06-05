# P16.9.2 Live Demo Terminal Runbook

This runbook contains only safe commands for the jury demonstration.

## 1. Start The Server Stack

```bash
cd experiments/qi-fl-ids-iot-final/deployment
docker compose -f docker-compose.final.yml --profile online --profile live-lab up -d --build
```

## 2. Open The Demo Dashboard

```text
http://192.168.56.1:8013/demo
```

## 3. VM1 Register

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_rpi_weak/run_node.py --server-url http://192.168.56.1:8020 --register
```

## 4. VM2 Register

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_smart_watch_medium/run_node.py --server-url http://192.168.56.1:8020 --register
```

## 5. VM1 Controlled PacketWindow Publish

```bash
cd ~/qi-fl-ids-iot
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py --broker 192.168.56.1 --node-id iot-drone-sitl --input-mode selected_12_scaled --window-size 30 --max-windows 1 --publish
```

## 6. VM2 Controlled PacketWindow Publish

```bash
cd ~/qi-fl-ids-iot
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py --broker 192.168.56.1 --node-id iot-smart-watch-medium --input-mode original_28_scaled --window-size 30 --max-windows 1 --publish
```

## 7. Optional Evidence Pages

```text
http://192.168.56.1:8013/api/live-lab/demo-state
http://192.168.56.1:8020/nodes
http://192.168.56.1:8020/assignments
http://192.168.56.1:3000
```

## Safety Reminder

No live capture, no lab scenario workstation, no attack traffic, no training, and no Flower process are used in this runbook.

