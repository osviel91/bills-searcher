# Phase 0B publication and recovery

## Publication contract

PostgreSQL owns each job's current `attempt_id` and `active_generation_id`. A
worker records a candidate generation, writes the document and passages under
generation-specific Elasticsearch IDs, refreshes and verifies both indexes,
then activates the generation in one PostgreSQL transaction conditional on the
current attempt and its unexpired lease. Search, filters, and evidence retrieval
read the active generation set from PostgreSQL. Elasticsearch data outside that
set is never a candidate result.

The previous generation remains available while a reindex is queued or running.
After activation it becomes an orphan. Elasticsearch cleanup is deferred for a
day and runs at worker startup and once per minute. Cleanup is serialized by a
PostgreSQL advisory transaction lock, processes at most 50 rows per batch,
skips active generations and live candidate owners, and retries partial deletes
idempotently. Existing indexed records without generation metadata remain
visible as legacy data until their next successful publication.

## Recovery

- If a worker exits, another worker requeues its job after the one-hour lease
  expires. The previous active generation remains queryable.
- If Elasticsearch writes fail before activation, the candidate remains
  invisible. Inspect the persisted `generations` row and `publication_errors`,
  then use the existing retry endpoint or MCP `reindex_invoices` operation.
- If a worker was stopped while candidate data was being written, it cannot
  activate after lease recovery. Candidate and orphan rows older than 24 hours
  with no current owner lease are deleted from both Elasticsearch indexes.
- To rebuild Elasticsearch, preserve PostgreSQL and the object store, recreate
  the indexes with the configured embedding dimensions, then requeue indexed
  documents with `reindex_invoices`. Processing re-extracts source files and
  recomputes embeddings; it requires the configured embedding service. Do not
  delete PostgreSQL or object-store volumes as part of index recovery.

## Residual limits

- Activation does not use a distributed transaction. A crash after PostgreSQL
  activation but before a response is observed is resolved by reading the
  active pointer; a crash before activation leaves only invisible candidate
  data.
- Cleanup is opportunistic: it runs at startup and on a 60-second worker timer,
  not in a standalone scheduler. If every worker is stopped, old generations
  remain until the next worker starts.
- A PostgreSQL connection failure is tested as a fail-closed search error, but
  recovery from a database-cluster outage or failover is not simulated.
- Legacy rows without a PostgreSQL generation pointer use the old active
  passage marker during reindex. Rebuild and reindex these rows before removing
  that compatibility path.
- `/api/health/live` reports process liveness. `/api/health/ready` reports
  PostgreSQL, Elasticsearch/index initialization, and object-storage readiness;
  `/api/health` retains its existing dependency details and includes aggregate
  queue/generation counters. No document content or exception message is
  returned as a worker error.

## Reproducible integration suite

Run `./scripts/run_integration.sh`. It creates a random Compose project with
isolated PostgreSQL, Elasticsearch, Azurite and settings volumes, starts the
API and two worker containers, uses a deterministic local embeddings stub, runs
the real MarkItDown extractor against generated PDF fixtures, and removes only
that project's containers, network and volumes on exit. Do not replace this
script's generated project name with a shared development Compose project when
using `down --volumes`.

The suite covers competing claims, expired-lease fencing, recovery and late
worker completion at extraction/index stages, failures after partial passages,
candidate document write, refresh and immediately before activation, a
PostgreSQL trigger failure during activation, repeated searches during
reindexing, legacy-index exclusion, PostgreSQL-read failure behavior, and
concurrent batched cleanup with an injected Elasticsearch delete failure.
