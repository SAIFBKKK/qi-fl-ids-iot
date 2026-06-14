#!/bin/bash
# P16.16 - Continuous UAV/MAVLink Observer
# Lancer une seule fois avant la soutenance.
# Reste actif jusqu'a Ctrl+C ou kill.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
AGENT="$REPO_ROOT/experiments/live_lab/nodes/iot_drone_sitl/mavlink_passive_agent.py"
LOG_FILE="$REPO_ROOT/experiments/qi-fl-ids-iot-final/outputs/reports/p1616_continuous_observer.jsonl"
MQTT_BROKER="192.168.56.1"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[P16.16] Starting continuous drone observer..."
echo "[P16.16] Log: $LOG_FILE"
echo "[P16.16] MQTT broker: $MQTT_BROKER"
echo "[P16.16] Press Ctrl+C to stop."
echo ""

python3 "$AGENT" \
  --scenario passive \
  --listen-port 14551 \
  --window-size 30 \
  --window-stride 15 \
  --mqtt-broker "$MQTT_BROKER" \
  --continuous \
  --status-interval 5 \
  --log-file "$LOG_FILE" \
  --log-level INFO
