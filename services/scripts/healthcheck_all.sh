#!/bin/bash
# healthcheck_all.sh - Verifie l'etat des services infra P1.
# Usage: ./services/scripts/healthcheck_all.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f services/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source services/.env
    set +a
fi

compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose --env-file services/.env -f services/docker-compose.yml "$@"
    else
        docker compose --env-file services/.env -f services/docker-compose.yml "$@"
    fi
}

check_http() {
    name="$1"
    url="$2"
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$response" = "200" ]; then
        echo "  OK $name"
        return 0
    fi
    echo "  FAIL $name (HTTP $response)"
    return 1
}

echo "=== Container status ==="
compose ps
echo ""

failures=0

echo "=== Infra health ==="
if docker exec mosquitto mosquitto_sub -h localhost \
    -u "${MQTT_USERNAME:-ids_user}" \
    -P "${MQTT_PASSWORD:-changeme_in_dotenv}" \
    -t '$SYS/broker/version' -C 1 -W 3 >/dev/null 2>&1; then
    echo "  OK mosquitto"
else
    echo "  FAIL mosquitto"
    failures=$((failures + 1))
fi

check_http "prometheus" "http://localhost:${PROMETHEUS_PORT:-9090}/-/healthy" || failures=$((failures + 1))
curl -s "http://localhost:${GRAFANA_PORT:-3000}/api/health" 2>/dev/null | grep -q '"database": "ok"' \
    && echo "  OK grafana" || { echo "  FAIL grafana"; failures=$((failures + 1)); }
check_http "mlflow" "http://localhost:${MLFLOW_PORT:-5000}/" || failures=$((failures + 1))

echo ""
if [ "$failures" -eq 0 ]; then
    echo "Summary: 4/4 healthy"
else
    echo "Summary: $((4 - failures))/4 healthy"
    exit 1
fi
