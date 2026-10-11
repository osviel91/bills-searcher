#!/bin/sh
set -eu

project="bills-shadow-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
compose="docker compose -f docker-compose.yml -f docker-compose.integration.yml -p $project"
cleanup() {
    $compose down --volumes --remove-orphans
}
trap cleanup EXIT INT TERM

export AS400_SHADOW_ENABLED=true
export INTEGRATION_TEST_PATTERN=test_shadow_integration.py
$compose up --build --scale worker=2 -d app worker

workers="$($compose ps -q worker)"
count=0
for container in $workers; do
    docker exec "$container" python -c 'import os; assert os.getenv("AS400_SHADOW_ENABLED") == "true"'
    count=$((count + 1))
done
test "$count" -eq 2

$compose run --rm --no-deps integration-tests
