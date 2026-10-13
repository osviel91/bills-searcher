from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"
LOCK_ID = 726412880


def apply(db):
    if not db.postgres:
        return
    db.execute("SELECT pg_advisory_xact_lock(?)", (LOCK_ID,))
    db.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    applied = {row["version"] for row in db.execute("SELECT version FROM schema_migrations")}
    files = sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
    known = {int(path.name[:3]) for path in files}
    if applied - known:
        raise RuntimeError(f"Database schema has unknown migration versions: {sorted(applied-known)}")
    for path in files:
        version = int(path.name[:3])
        if version in applied:
            continue
        db.execute(path.read_text())
        db.execute("INSERT INTO schema_migrations(version,name) VALUES(?,?)", (version, path.stem))
