import os
import sqlite3
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL", "")
_initialized = set()


class Connection:
    def __init__(self, raw, postgres):
        self.raw = raw
        self.postgres = postgres

    def execute(self, query, params=()):
        return self.raw.execute(query.replace("?", "%s") if self.postgres else query, params)

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
    db = Connection(raw, postgres)
    key = DATABASE_URL if postgres else str(db_path)
    if key not in _initialized:
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, filename TEXT, object_key TEXT, status TEXT, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, locked_at TIMESTAMP)")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at DESC)")
        db.commit()
        _initialized.add(key)
    return db


def claim_next_job(path: Path | None = None):
    with connect(path) as db:
        if db.postgres:
            row = db.execute("""WITH candidate AS (
                SELECT id FROM jobs WHERE status='queued' ORDER BY created_at,id
                FOR UPDATE SKIP LOCKED LIMIT 1
            ) UPDATE jobs SET status='processing',locked_at=CURRENT_TIMESTAMP
              WHERE id=(SELECT id FROM candidate) RETURNING *""").fetchone()
        else:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='processing',locked_at=CURRENT_TIMESTAMP WHERE id=?", (row["id"],))
                row = dict(row)
            db.commit()
        return dict(row) if row else None


def recover_stale_jobs(path: Path | None = None):
    with connect(path) as db:
        if db.postgres:
            db.execute("UPDATE jobs SET status='queued',error='Worker lease expired; job requeued',locked_at=NULL WHERE status='processing' AND locked_at < CURRENT_TIMESTAMP - INTERVAL '1 hour'")
        else:
            db.execute("UPDATE jobs SET status='queued',error='Worker lease expired; job requeued',locked_at=NULL WHERE status='processing' AND locked_at < datetime('now','-1 hour')")
        db.commit()
