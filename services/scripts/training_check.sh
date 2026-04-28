#!/usr/bin/env bash
# Validate Mode B training profile runtime for QI-FL-IDS-IoT.
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
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --profile training "$@"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        pass "$1 available"
        return 0
    fi
    fail "$1 not available"
    return 1
}

http_ok() {
    url="$1"
    curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 "$url" 2>/dev/null || printf '000'
}

container_state() {
    docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$1" 2>/dev/null || true
}

read_env_value() {
    key="$1"
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi
    sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r'
}

check_running() {
    name="$1"
    state="$(container_state "$name")"
    status="${state%% *}"
    if [ "$status" = "running" ]; then
        pass "container running: $name"
    else
        fail "container not running: $name"
    fi
}

check_running_or_exited_zero() {
    name="$1"
    state="$(container_state "$name")"
    status="${state%% *}"
    exit_code="${state##* }"
    if [ "$status" = "running" ]; then
        pass "container running: $name"
    elif [ "$status" = "exited" ] && [ "$exit_code" = "0" ]; then
        pass "container exited cleanly: $name"
    else
        fail "container not healthy: $name (state=${state:-missing})"
    fi
}

printf '=== QI-FL-IDS-IoT Mode B training check ===\n'
printf 'Repo root: %s\n' "$REPO_ROOT"

if ! check_command docker; then
    printf '\nMode B training check: FAIL (%s failure(s))\n' "$FAILURES"
    exit 1
fi

if docker compose --env-file "$ENV_EXAMPLE" -f "$COMPOSE_FILE" --profile training config --quiet >/dev/null 2>&1; then
    pass "docker compose training config"
else
    fail "docker compose training config"
fi

if [ ! -f "$ENV_FILE" ]; then
    fail "Create services/.env from .env.example first"
    printf '\nMode B training check: FAIL (%s failure(s))\n' "$FAILURES"
    exit 1
fi

TRAINING_MODE="$(read_env_value TRAINING_MODE)"
TRAINING_MODE="${TRAINING_MODE:-mock}"
pass "training mode: $TRAINING_MODE"

if [ "$TRAINING_MODE" = "real" ]; then
    check_running_or_exited_zero fl-server
else
    check_running fl-server
fi
check_running_or_exited_zero fl-client-1
check_running_or_exited_zero fl-client-2
check_running_or_exited_zero fl-client-3

if check_command curl; then
    code="$(http_ok "http://localhost:5000")"
    if [ "$code" = "200" ]; then
        pass "mlflow reachable"
    else
        fail "mlflow reachable (HTTP $code)"
    fi
fi

if [ "$TRAINING_MODE" = "real" ]; then
    log_pattern='TRAINING_MODE=real|run_experiment|exp_v4_multitier|scientific runner'
else
    log_pattern='training|round|Flower|FedAvg'
fi

if docker logs fl-server 2>&1 | grep -Eiq "$log_pattern"; then
    pass "fl-server logs contain training markers"
else
    fail "fl-server logs contain training markers"
fi

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
    printf 'Mode B training check: PASS\n'
    exit 0
fi

printf 'Mode B training check: FAIL (%s failure(s))\n' "$FAILURES"
exit 1
