import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import jobstore
import migrate


class MigrationTests(unittest.TestCase):
    def test_moves_legacy_pdf_and_requeues_interrupted_job(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jobstore, "DATABASE_URL", ""):
            root = Path(temp)
            legacy = root / "jobs.sqlite3"
            (root / "documents").mkdir()
            pdf_path = root / "documents" / "old.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            with closing(sqlite3.connect(legacy)) as db:
                db.execute("CREATE TABLE jobs (id TEXT, sha256 TEXT, filename TEXT, path TEXT, status TEXT, error TEXT, created_at TEXT)")
                db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("old", "hash", "old.pdf", "/data/documents/old.pdf", "processing", None, "2024-01-01 00:00:00"))
                db.commit()

            target_db = root / "new.sqlite3"
            es = Mock()
            es.indices.exists.return_value = False
            with patch.object(migrate, "LEGACY_DB", legacy), patch.object(migrate, "connect", side_effect=lambda: jobstore.connect(target_db)), patch.object(migrate, "ensure_container"), patch.object(migrate, "put") as put, patch.object(migrate, "elasticsearch", return_value=es):
                migrate.main()

            with closing(jobstore.connect(target_db)) as db:
                row = db.execute("SELECT status,object_key FROM jobs WHERE id='old'").fetchone()
            self.assertEqual(row["status"], "queued")
            self.assertEqual(row["object_key"], "documents/old.pdf")
            put.assert_called_once_with("documents/old.pdf", b"%PDF-1.4")


if __name__ == "__main__":
    unittest.main()
