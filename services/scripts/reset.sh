#!/bin/bash
# reset.sh - Reset propre du stack microservices.
# Usage: ./services/scripts/reset.sh [--hard]
set -euo pipefail

cd "$(dirname "$0")/../.."

compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose --env-file services/.env -f services/docker-compose.yml "$@"
    else
        docker compose --env-file services/.env -f services/docker-compose.yml "$@"
    fi
}

if [ "${1:-}" = "--hard" ]; then
    echo "HARD RESET: suppression containers + volumes Docker + bind mounts data"
    read -r -p "Confirmer ? [y/N] " confirm
    if [ "$confirm" != "y" ]; then
        echo "Annule"
        exit 0
    fi

    compose --profile orchestration --profile preprocessing --profile training down -v 2>/dev/null || true
    rm -rf outputs/mlruns/* outputs/logs/* outputs/checkpoints/* 2>/dev/null || true
    echo "Hard reset done"
    echo "Restart with: docker-compose --env-file services/.env -f services/docker-compose.yml up -d"
else
    echo "Soft reset: arret containers, conservation des volumes"
    compose --profile orchestration --profile preprocessing --profile training down 2>/dev/null || true
    echo "Soft reset done"
    echo "Restart with: docker-compose --env-file services/.env -f services/docker-compose.yml up -d"
fi
