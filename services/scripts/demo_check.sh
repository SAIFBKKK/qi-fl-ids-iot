#!/usr/bin/env bash
# Validate Mode A runtime for QI-FL-IDS-IoT.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SERVICES_DIR/.." && pwd)"
COMPOSE_FILE="$SERVICES_DIR/docker-compose.yml"
ENV_EXAMPLE="$SERVICES_DIR/.env.example"
ENV_FILE="$SERVICES_DIR/.env"

FAILURES=0

pass() {
    printf 'PASS %s\n' "$1"
}

fail() {
    printf 'FAIL %s\n' "$1"
    FAILURES=$((FAILURES + 1))
}

compose() {
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        pass "$1 available"
        return 0
    fi
    fail "$1 not available"
    return 1
}

http_body() {
    url="$1"
    curl --silent --show-error --max-time 5 "$url" 2>/dev/null
}

check_http_contains() {
    name="$1"
    url="$2"
    pattern="$3"
    body="$(http_body "$url")"
    if printf '%s' "$body" | grep -q "$pattern"; then
        pass "$name"
    else
        fail "$name ($url)"
    fi
}

check_http_status() {
    name="$1"
    url="$2"
    code="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 "$url" 2>/dev/null || printf '000')"
    if [ "$code" = "200" ]; then
        pass "$name"
    else
        fail "$name ($url returned HTTP $code)"
    fi
}

printf '=== QI-FL-IDS-IoT Mode A demo check ===\n'
printf 'Repo root: %s\n' "$REPO_ROOT"

check_command docker
check_command curl

if docker compose --env-file "$ENV_EXAMPLE" -f "$COMPOSE_FILE" config --quiet >/dev/null 2>&1; then
    pass "docker compose config"
else
    fail "docker compose config"
fi

if [ ! -f "$ENV_FILE" ]; then
    fail "Create services/.env from .env.example first"
    printf '\n'
    printf 'Summary: %s failure(s)\n' "$FAILURES"
    exit 1
fi

expected_services="mosquitto iot-node-1 traffic-generator prometheus grafana mlflow"
running_services="$(compose ps --services --filter status=running 2>/dev/null || true)"
for service in $expected_services; do
    if printf '%s\n' "$running_services" | grep -qx "$service"; then
        pass "container running: $service"
    else
        fail "container not running: $service"
    fi
done

if command -v curl >/dev/null 2>&1; then
    check_http_contains "iot-node health" "http://localhost:8001/health" '"status":"ok"\|"status": "ok"'
    check_http_contains "iot-node ready" "http://localhost:8001/ready" '"ready":true\|"ready": true'
    check_http_contains "traffic-generator health" "http://localhost:8010/health" '"status":"ok"\|"status": "ok"'
    check_http_contains "traffic-generator ready" "http://localhost:8010/ready" '"ready":true\|"ready": true'
    check_http_status "prometheus ready" "http://localhost:9090/-/ready"
    check_http_contains "grafana health" "http://localhost:3000/api/health" '"database":"ok"\|"database": "ok"'
    check_http_status "mlflow" "http://localhost:5000"
    check_http_contains "iot-node metrics" "http://localhost:8001/metrics" "ids_flows_received_total"
    check_http_contains "traffic-generator metrics" "http://localhost:8010/metrics" "traffic_generator_flows_published_total"
fi

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
    printf 'Mode A demo check: PASS\n'
    exit 0
fi

printf 'Mode A demo check: FAIL (%s failure(s))\n' "$FAILURES"
exit 1
