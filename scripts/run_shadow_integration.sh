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

app_container="$($compose ps -q app)"
attempt=0
while test "$(docker inspect --format '{{.State.Health.Status}}' "$app_container")" != healthy; do
    attempt=$((attempt + 1))
    if test "$attempt" -ge 60; then
        $compose logs app schema-migrate
        exit 1
    fi
    sleep 1
done
docker exec "$app_container" python -c 'import os; assert "DATABASE_MIGRATION_URL" not in os.environ and "DATABASE_ADMIN_URL" not in os.environ'

workers="$($compose ps -q worker)"
count=0
for container in $workers; do
    docker exec "$container" python -c 'import os; assert os.getenv("AS400_SHADOW_ENABLED") == "true"; assert "DATABASE_MIGRATION_URL" not in os.environ and "DATABASE_ADMIN_URL" not in os.environ'
    count=$((count + 1))
done
test "$count" -eq 2

$compose run --rm --no-deps integration-tests
