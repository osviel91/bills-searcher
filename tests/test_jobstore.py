import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import jobstore


class QueueTests(unittest.TestCase):
    def test_claims_each_job_once_and_recovers_expired_worker_lease(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jobstore, "DATABASE_URL", ""):
            db_path = Path(temp) / "jobs.sqlite3"
            jobstore._initialized.clear()
            with closing(jobstore.connect(db_path)) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("one", "a", "a.pdf", "a", "queued"))
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("two", "b", "b.pdf", "b", "queued"))
                db.commit()

            self.assertEqual(jobstore.claim_next_job(db_path)["id"], "one")
            self.assertEqual(jobstore.claim_next_job(db_path)["id"], "two")
            self.assertIsNone(jobstore.claim_next_job(db_path))
            with closing(jobstore.connect(db_path)) as db:
                db.execute("UPDATE jobs SET locked_at='2000-01-01 00:00:00' WHERE id='one'")
                db.commit()
            jobstore.recover_stale_jobs(db_path)
            self.assertEqual(jobstore.claim_next_job(db_path)["id"], "one")
