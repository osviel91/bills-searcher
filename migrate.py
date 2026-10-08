import os
import sqlite3
from contextlib import closing
from pathlib import Path

from jobstore import connect
from object_store import ensure_container, put
from elasticsearch.helpers import scan
from storage import INDEX, client as elasticsearch

LEGACY_DB = Path(os.getenv("LEGACY_DB", "/legacy/jobs.sqlite3"))


def main():
    ensure_container()
    with closing(connect()) as target:
        if not LEGACY_DB.is_file():
            print("No legacy SQLite database; initialized PostgreSQL and object storage.")
            return
        with closing(sqlite3.connect(LEGACY_DB)) as legacy:
            legacy.row_factory = sqlite3.Row
            rows = legacy.execute("SELECT id,sha256,filename,path,status,error,created_at FROM jobs").fetchall()
        migrated = 0
        for row in rows:
            existing = target.execute("SELECT id FROM jobs WHERE id=? OR sha256=?", (row["id"], row["sha256"])).fetchone()
            if existing:
                continue
            key = f"documents/{row['id']}.pdf"
            source = Path(row["path"])
            if not source.is_file():
                source = LEGACY_DB.parent / "documents" / Path(row["path"]).name
            if source.is_file():
                put(key, source.read_bytes())
            status = "queued" if row["status"] == "processing" else row["status"]
            target.execute("""INSERT INTO jobs(id,sha256,filename,object_key,status,error,created_at)
                VALUES(?,?,?,?,?,?,?)""", (row["id"], row["sha256"], row["filename"],
                key if source.is_file() else None, status, row["error"], row["created_at"]))
            target.commit()
            migrated += 1
        requeued = 0
        es = elasticsearch()
        if es.indices.exists(index=INDEX):
            for hit in scan(es, index=INDEX, query={"query": {"bool": {"must_not": [{"exists": {"field": "content"}}]}}}, _source=False):
                result = target.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=? AND status='indexed'", (hit["_id"],))
                requeued += result.rowcount
            target.commit()
        print(f"Migrated {migrated} legacy jobs; requeued {requeued} legacy index entries.")


if __name__ == "__main__":
    main()
