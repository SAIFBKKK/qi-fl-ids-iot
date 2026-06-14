#!/bin/bash
# Genere le fichier mosquitto/passwords a partir de services/.env.
# Usage: ./services/scripts/generate_mqtt_password.sh
set -euo pipefail

SERVICES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$SERVICES_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Copy services/.env.example to services/.env first."
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ -z "${MQTT_USERNAME:-}" ] || [ -z "${MQTT_PASSWORD:-}" ]; then
    echo "Error: MQTT_USERNAME and MQTT_PASSWORD must be set in $ENV_FILE."
    exit 1
fi

if command -v cygpath >/dev/null 2>&1; then
    DOCKER_SERVICES_DIR="$(cygpath -w "$SERVICES_DIR")"
else
    DOCKER_SERVICES_DIR="$SERVICES_DIR"
fi

MSYS_NO_PATHCONV=1 docker run --rm -v "$DOCKER_SERVICES_DIR/mosquitto:/mosquitto/config" \
    eclipse-mosquitto:2.0.18 \
    mosquitto_passwd -b -c /mosquitto/config/passwords \
    "$MQTT_USERNAME" "$MQTT_PASSWORD"

echo "Password file generated at services/mosquitto/passwords"
