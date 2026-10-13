import os
import sqlite3
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from infrastructure.migrations import apply as apply_migrations

DATABASE_URL = os.getenv("DATABASE_URL", "")
LEASE_SECONDS = max(1, int(os.getenv("JOB_LEASE_SECONDS", "3600")))
HEARTBEAT_SECONDS = max(0.1, float(os.getenv("JOB_HEARTBEAT_SECONDS", "30")))
GENERATION_RETENTION_SECONDS = max(1, int(os.getenv("GENERATION_RETENTION_SECONDS", "86400")))
_initialized = set()


class Connection:
    def __init__(self, raw, postgres):
        self.raw = raw
        self.postgres = postgres

    def execute(self, query, params=()):
        if self.postgres and isinstance(query, str):
            query = query.replace("?", "%s")
        return self.raw.execute(query, params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.raw.rollback() if exc_type else self.raw.commit()
        finally:
            self.raw.close()

    def __getattr__(self, name):
        return getattr(self.raw, name)


def connect(path: Path | None = None):
    postgres = bool(DATABASE_URL)
    db_path = path or Path(os.getenv("JOBS_DB", "/tmp/invoice-jobs.sqlite3"))
    if not postgres:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (psycopg.connect(DATABASE_URL, row_factory=dict_row) if postgres else
           sqlite3.connect(db_path, timeout=30))
    if not postgres:
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys=ON")
    db = Connection(raw, postgres)
    key = DATABASE_URL if postgres else str(db_path)
    initialize_postgres = os.getenv("SCHEMA_INITIALIZER") == "true"
    if key not in _initialized and (not postgres or initialize_postgres):
        if postgres:
            db.execute("SELECT pg_advisory_xact_lock(?)", (726412880,))
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT, object_key TEXT, status TEXT, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, locked_at TIMESTAMP)")
        if postgres:
            db.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS attempt_id TEXT")
        elif "attempt_id" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)") }:
            db.execute("ALTER TABLE jobs ADD COLUMN attempt_id TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at DESC)")
        for column, definition in (("active_generation_id", "TEXT"), ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                                  ("lease_expirations", "INTEGER NOT NULL DEFAULT 0"),
                                  ("publication_errors", "INTEGER NOT NULL DEFAULT 0"),
                                  ("processing_started_at", "TIMESTAMP"),
                                  ("processing_duration_ms", "INTEGER")):
            if postgres:
                db.execute(f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {column} {definition}")
            elif column not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
        db.execute("""CREATE TABLE IF NOT EXISTS generations (
            generation_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
            status TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP, passage_count INTEGER NOT NULL DEFAULT 0)""")
        db.execute("CREATE INDEX IF NOT EXISTS generations_job_status ON generations(job_id,status)")
        db.execute("CREATE TABLE IF NOT EXISTS maintenance_metrics (name TEXT PRIMARY KEY,value INTEGER NOT NULL DEFAULT 0)")
        db.execute("""CREATE TABLE IF NOT EXISTS extraction_runs (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES jobs(id),
            extractor_version TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed','discarded_fenced')),
            warnings_json TEXT NOT NULL, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        db.execute("CREATE INDEX IF NOT EXISTS extraction_runs_document ON extraction_runs(document_id,created_at)")
        db.execute("""CREATE TABLE IF NOT EXISTS extraction_pages (
            run_id TEXT NOT NULL REFERENCES extraction_runs(id), page_number INTEGER NOT NULL,
            structure_json TEXT NOT NULL, PRIMARY KEY(run_id,page_number))""")
        db.execute("""CREATE TABLE IF NOT EXISTS extraction_fields (
            run_id TEXT NOT NULL REFERENCES extraction_runs(id), name TEXT NOT NULL,
            value_json TEXT, warnings_json TEXT NOT NULL, PRIMARY KEY(run_id,name))""")
        db.execute("""CREATE TABLE IF NOT EXISTS extraction_lines (
            run_id TEXT NOT NULL REFERENCES extraction_runs(id), line_number INTEGER NOT NULL,
            description TEXT, values_json TEXT NOT NULL, warnings_json TEXT NOT NULL,
            PRIMARY KEY(run_id,line_number))""")
        db.execute("""CREATE TABLE IF NOT EXISTS extraction_evidence (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, field_name TEXT, line_number INTEGER,
            page_number INTEGER NOT NULL, block_id TEXT, quote TEXT NOT NULL, bounds_json TEXT,
            CHECK((field_name IS NOT NULL AND line_number IS NULL) OR
                   (field_name IS NULL AND line_number IS NOT NULL)),
            FOREIGN KEY(run_id,field_name) REFERENCES extraction_fields(run_id,name),
            FOREIGN KEY(run_id,line_number) REFERENCES extraction_lines(run_id,line_number),
            FOREIGN KEY(run_id,page_number) REFERENCES extraction_pages(run_id,page_number))""")
        if not postgres:
            for table, columns in {
                "extraction_runs": ("document_sha256 TEXT", "attempt_id TEXT", "detector_id TEXT",
                    "detector_version TEXT", "template TEXT", "extractor_id TEXT",
                    "extractor_set_fingerprint TEXT", "execution_mode TEXT", "result_schema_version INTEGER",
                    "idempotency_key TEXT", "abstention_reason TEXT", "diagnostics_code TEXT",
                    "result_fingerprint TEXT", "result_fingerprint_version INTEGER NOT NULL DEFAULT 1",
                    "finished_at TIMESTAMP", "execution_slot TEXT", "document_size_bytes INTEGER"),
                "extraction_fields": ("observation_status TEXT", "candidates_json TEXT"),
                "extraction_evidence": ("label_quote TEXT", "label_bounds_json TEXT", "method TEXT",
                    "extractor_id TEXT", "extractor_version TEXT", "source_token_ids_json TEXT"),
            }.items():
                existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                for definition in columns:
                    if definition.split()[0] not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS extraction_runs_document_idempotency "
                       "ON extraction_runs(document_id,idempotency_key) WHERE idempotency_key IS NOT NULL")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS extraction_runs_document_attempt_slot "
                       "ON extraction_runs(document_id,attempt_id,execution_slot) "
                       "WHERE attempt_id IS NOT NULL AND execution_slot IS NOT NULL")
        db.commit()
        if postgres:
            apply_migrations(db)
            db.commit()
        _initialized.add(key)
    return db


def claim_next_job(path: Path | None = None):
    recover_stale_jobs(path)
    attempt_id = str(uuid.uuid4())
    with connect(path) as db:
        if db.postgres:
            row = db.execute("""WITH candidate AS (
                SELECT id FROM jobs WHERE status='queued' ORDER BY created_at,id
                FOR UPDATE SKIP LOCKED LIMIT 1
            ) UPDATE jobs SET status='processing',locked_at=CURRENT_TIMESTAMP,processing_started_at=CURRENT_TIMESTAMP,
                attempt_count=attempt_count+1,attempt_id=?
              WHERE id=(SELECT id FROM candidate) RETURNING *""", (attempt_id,)).fetchone()
        else:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row:
                db.execute("""UPDATE jobs SET status='processing',locked_at=CURRENT_TIMESTAMP,
                            processing_started_at=CURRENT_TIMESTAMP,attempt_count=attempt_count+1,
                            attempt_id=? WHERE id=?""", (attempt_id, row["id"]))
                row = dict(row)
                row["attempt_id"] = attempt_id
            db.commit()
        return dict(row) if row else None


def recover_stale_jobs(path: Path | None = None):
    with connect(path) as db:
        if db.postgres:
            stale = f"CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'"
            db.execute(f"""UPDATE jobs SET status='queued',error='Worker lease expired; job requeued',
                        locked_at=NULL,attempt_id=NULL,lease_expirations=lease_expirations+1
                        WHERE status='processing' AND (locked_at IS NULL OR locked_at < {stale})""")
        else:
            db.execute(f"""UPDATE jobs SET status='queued',error='Worker lease expired; job requeued',
                        locked_at=NULL,attempt_id=NULL,lease_expirations=lease_expirations+1
                        WHERE status='processing' AND (locked_at IS NULL OR locked_at < datetime('now','-{LEASE_SECONDS} seconds'))""")
        db.commit()


def heartbeat_job(job_id: str, attempt_id: str, path: Path | None = None) -> bool:
    with connect(path) as db:
        lease = (f"locked_at >= CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'" if db.postgres else
                 f"locked_at >= datetime('now','-{LEASE_SECONDS} seconds')")
        result = db.execute(f"UPDATE jobs SET locked_at=CURRENT_TIMESTAMP WHERE id=? AND status='processing' "
                            f"AND attempt_id=? AND {lease}", (job_id, attempt_id))
        return result.rowcount == 1


def prepare_generation(job_id: str, attempt_id: str, generation_id: str, passage_count: int,
                       path: Path | None = None) -> bool:
    with connect(path) as db:
        lease = (f"locked_at >= CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'" if db.postgres else
                 f"locked_at >= datetime('now','-{LEASE_SECONDS} seconds')")
        result = db.execute(f"""INSERT INTO generations(generation_id,job_id,attempt_id,status,passage_count)
            SELECT ?,id,?,'candidate',? FROM jobs
            WHERE id=? AND status='processing' AND attempt_id=? AND {lease}""",
            (generation_id, attempt_id, passage_count, job_id, attempt_id))
        return result.rowcount == 1


def activate_generation(job_id: str, attempt_id: str, generation_id: str,
                         path: Path | None = None) -> bool:
    with connect(path) as db:
        if db.postgres:
            lease = f"locked_at >= CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'"
            duration = "(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - processing_started_at))*1000)::INTEGER"
        else:
            lease = f"locked_at >= datetime('now','-{LEASE_SECONDS} seconds')"
            duration = "CAST((julianday('now')-julianday(processing_started_at))*86400000 AS INTEGER)"
        result = db.execute(f"""UPDATE jobs SET status='indexed',error=NULL,locked_at=NULL,
            active_generation_id=?,processing_duration_ms={duration} WHERE id=? AND status='processing' AND attempt_id=? AND {lease}
            AND EXISTS (SELECT 1 FROM generations WHERE generation_id=? AND job_id=? AND attempt_id=? AND status='candidate')""",
            (generation_id, job_id, attempt_id, generation_id, job_id, attempt_id))
        if result.rowcount != 1:
            return False
        db.execute("UPDATE generations SET status='orphaned' WHERE job_id=? AND status='active'", (job_id,))
        db.execute("UPDATE generations SET status='active',published_at=CURRENT_TIMESTAMP WHERE generation_id=?",
                   (generation_id,))
        return True


def active_generations(path: Path | None = None) -> list[str]:
    with connect(path) as db:
        return [row["active_generation_id"] for row in db.execute(
            "SELECT active_generation_id FROM jobs WHERE status='indexed' AND active_generation_id IS NOT NULL")]


def visible_generation_query(*, passages: bool = False, path: Path | None = None) -> dict:
    with connect(path) as db:
        rows = list(db.execute("SELECT id,active_generation_id FROM jobs WHERE active_generation_id IS NOT NULL OR status='indexed'"))
    generations = [row["active_generation_id"] for row in rows
                   if row["active_generation_id"] and row["active_generation_id"] != "legacy"]
    legacy_ids = [row["id"] for row in rows
                  if not row["active_generation_id"] or row["active_generation_id"] == "legacy"]
    alternatives = []
    if generations:
        alternatives.append({"terms": {"generation": generations}})
    if legacy_ids:
        filters = [{"terms": {"invoice_id": legacy_ids}}]
        if passages:
            filters.append({"term": {"active": True}})
        else:
            alternatives.append({"bool": {"filter": filters,
                                           "must_not": [{"exists": {"field": "generation"}}]}})
        if passages:
            alternatives.append({"bool": {"filter": filters}})
    return {"bool": {"should": alternatives, "minimum_should_match": 1}} if alternatives else {"match_none": {}}


def processing_metrics(path: Path | None = None) -> dict:
    with connect(path) as db:
        statuses = {row["status"]: row["count"] for row in db.execute(
            "SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")}
        generations = {row["status"]: row["count"] for row in db.execute(
            "SELECT status,COUNT(*) AS count FROM generations GROUP BY status")}
        totals = db.execute("""SELECT COALESCE(SUM(CASE WHEN attempt_count > 0 THEN attempt_count-1 ELSE 0 END),0) AS retries,
            COALESCE(SUM(lease_expirations),0) AS expired_leases,
            COALESCE(SUM(publication_errors),0) AS publication_errors,
            COALESCE(AVG(processing_duration_ms),0) AS average_processing_ms FROM jobs""").fetchone()
        maintenance = {row["name"]: row["value"] for row in db.execute(
            "SELECT name,value FROM maintenance_metrics")}
        return {"jobs": statuses, "generations": generations, "maintenance": maintenance,
                **{key: (int(value) if key == "average_processing_ms" else value)
                   for key, value in dict(totals).items()}}


def record_publication_error(job_id: str, attempt_id: str, path: Path | None = None):
    with connect(path) as db:
        db.execute("UPDATE jobs SET publication_errors=publication_errors+1 WHERE id=? AND status='processing' AND attempt_id=?",
                   (job_id, attempt_id))


def lock_generation_cleanup(db) -> bool:
    if db.postgres:
        return db.execute("SELECT pg_try_advisory_xact_lock(742901631) AS locked").fetchone()["locked"]
    db.execute("BEGIN IMMEDIATE")
    return True


def stale_generations(db, batch_size: int = 50) -> list[dict]:
    age = (f"CURRENT_TIMESTAMP - INTERVAL '{GENERATION_RETENTION_SECONDS} seconds'" if db.postgres else
           f"datetime('now','-{GENERATION_RETENTION_SECONDS} seconds')")
    lease = (f"j.locked_at >= CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'" if db.postgres else
             f"j.locked_at >= datetime('now','-{LEASE_SECONDS} seconds')")
    locking = " FOR UPDATE OF g SKIP LOCKED" if db.postgres else ""
    return [dict(row) for row in db.execute(f"""SELECT g.generation_id,g.job_id FROM generations g
        WHERE g.status IN ('candidate','orphaned') AND g.created_at < {age}
        AND NOT EXISTS (SELECT 1 FROM jobs active WHERE active.active_generation_id=g.generation_id)
        AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.id=g.job_id AND j.status='processing'
            AND j.attempt_id=g.attempt_id AND {lease})
        ORDER BY g.created_at,g.generation_id LIMIT ?{locking}""", (batch_size,))]


def delete_generation(db, generation_id: str) -> bool:
    result = db.execute("""DELETE FROM generations WHERE generation_id=? AND status IN ('candidate','orphaned')
                         AND NOT EXISTS (SELECT 1 FROM jobs WHERE active_generation_id=?)""",
                        (generation_id, generation_id))
    return result.rowcount == 1


def record_cleanup_metrics(db, deleted: int, errors: int):
    for name, count in (("generations_cleaned", deleted), ("generation_cleanup_errors", errors)):
        if count:
            db.execute("""INSERT INTO maintenance_metrics(name,value) VALUES(?,?)
                ON CONFLICT(name) DO UPDATE SET value=maintenance_metrics.value+excluded.value""",
                       (name, count))
