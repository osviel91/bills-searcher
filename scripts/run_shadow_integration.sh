#!/bin/sh
set -eu

keep_on_failure=false
case "${1:-}" in
    "") ;;
    --keep-on-failure) keep_on_failure=true ;;
    *) printf 'Usage: %s [--keep-on-failure]\n' "$0" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    printf 'Usage: %s [--keep-on-failure]\n' "$0" >&2
    exit 2
fi

project="bills-shadow-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
azurite_credentials="$(python3 -c 'import base64, secrets, uuid; print(f"bills{uuid.uuid4().hex[:12]}:{base64.b64encode(secrets.token_bytes(64)).decode()}")')"
AZURITE_ACCOUNT_NAME=${azurite_credentials%%:*}
AZURITE_ACCOUNT_KEY=${azurite_credentials#*:}
export AZURITE_ACCOUNT_NAME AZURITE_ACCOUNT_KEY
unset azurite_credentials
compose="docker compose --env-file .env --env-file .env.migration -f docker-compose.yml -f docker-compose.integration.yml -p $project"
artifact_dir=${BILLS_TEST_ARTIFACTS_DIR:-"$HOME/Library/Application Support/BillsSearcher/test-runs"}
(umask 077 && mkdir -p "$artifact_dir")
log_file="$artifact_dir/$project.log"
(umask 077 && : > "$log_file")

cleanup() {
    result=$?
    trap - EXIT INT TERM
    $compose logs --no-color >> "$log_file" 2>&1 || true
    if [ "$result" -ne 0 ] && [ "$keep_on_failure" = true ]; then
        printf 'project retained after failure: %s\nlog: %s\n' "$project" "$log_file"
        printf 'cleanup command: %s down --rmi local --volumes --remove-orphans\n' "$compose"
        exit "$result"
    fi
    if $compose down --rmi local --volumes --remove-orphans >> "$log_file" 2>&1; then
        printf 'project=%s exit=%s cleanup=complete log=%s\n' "$project" "$result" "$log_file"
    else
        printf 'cleanup incomplete; resources may remain for project %s\nlog: %s\n' "$project" "$log_file" >&2
        printf 'cleanup command: %s down --rmi local --volumes --remove-orphans\n' "$compose" >&2
        result=1
    fi
    exit "$result"
}
trap cleanup EXIT INT TERM
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'project=%s\n' "$project" > "$log_file"
printf 'project=%s log=%s\n' "$project" "$log_file"
export AS400_SHADOW_ENABLED=true
export AS400_SHADOW_PERSISTENCE_ENABLED=false
export INTEGRATION_TEST_PATTERN=test_shadow_integration.py
$compose up --build --scale worker=2 -d app worker >> "$log_file" 2>&1

app_container="$($compose ps -q app)"
attempt=0
while test "$(docker inspect --format '{{.State.Health.Status}}' "$app_container")" != healthy; do
    attempt=$((attempt + 1))
    if test "$attempt" -ge 60; then
        $compose logs --no-color app schema-migrate >> "$log_file" 2>&1 || true
        exit 1
    fi
    sleep 1
done
docker exec "$app_container" python -c 'import os; assert "DATABASE_MIGRATION_URL" not in os.environ and "DATABASE_ADMIN_URL" not in os.environ; assert os.getenv("AS400_SHADOW_PERSISTENCE_ENABLED") == "false"'

workers="$($compose ps -q worker)"
count=0
for container in $workers; do
    docker exec "$container" python -c 'import os; assert os.getenv("AS400_SHADOW_ENABLED") == "true"; assert os.getenv("AS400_SHADOW_PERSISTENCE_ENABLED") == "false"; assert "DATABASE_MIGRATION_URL" not in os.environ and "DATABASE_ADMIN_URL" not in os.environ'
    count=$((count + 1))
done
test "$count" -eq 2

if $compose run --build --rm --no-deps integration-tests >> "$log_file" 2>&1; then
    test_status=0
else
    test_status=$?
fi
printf 'test_exit=%s\n' "$test_status" >> "$log_file"
exit "$test_status"
