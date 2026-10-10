#!/bin/sh
set -eu

project="bills-phase0c-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
compose="docker compose -f docker-compose.yml -f docker-compose.integration.yml -p $project"
cleanup() {
    $compose down --volumes --remove-orphans
}
trap cleanup EXIT INT TERM
$compose up --build --scale worker=2 --abort-on-container-exit --exit-code-from integration-tests integration-tests
