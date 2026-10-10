import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import jobstore


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.database_url = patch.object(jobstore, "DATABASE_URL", "")
        self.database_url.start()
        jobstore._initialized.clear()

    def tearDown(self):
        self.database_url.stop()

    def test_claims_each_job_once_and_recovers_expired_worker_lease(self):
        with tempfile.TemporaryDirectory() as temp:
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
            recovered = jobstore.claim_next_job(db_path)
            self.assertEqual(recovered["id"], "one")
            self.assertTrue(recovered["attempt_id"])
            self.assertTrue(jobstore.heartbeat_job("one", recovered["attempt_id"], db_path))
            self.assertFalse(jobstore.heartbeat_job("one", "stale-attempt", db_path))

    def test_generation_activation_is_fenced_and_preserves_previous_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "jobs.sqlite3"
            with jobstore.connect(db_path) as db:
                db.execute("INSERT INTO jobs(id,sha256,status,attempt_id,active_generation_id,locked_at) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                           ("doc", "hash", "processing", "attempt-2", "generation-1"))
                db.execute("INSERT INTO generations(generation_id,job_id,attempt_id,status) VALUES(?,?,?,'active')",
                           ("generation-1", "doc", "attempt-1"))
                db.commit()
            self.assertTrue(jobstore.prepare_generation("doc", "attempt-2", "generation-2", 1, db_path))
            self.assertFalse(jobstore.activate_generation("doc", "attempt-1", "generation-1", db_path))
            with jobstore.connect(db_path) as db:
                active = db.execute("SELECT active_generation_id,status FROM jobs WHERE id='doc'").fetchone()
            self.assertEqual((active["active_generation_id"], active["status"]), ("generation-1", "processing"))
            visible = jobstore.visible_generation_query(path=db_path)
            self.assertEqual(visible["bool"]["should"][0], {"terms": {"generation": ["generation-1"]}})
            self.assertTrue(jobstore.activate_generation("doc", "attempt-2", "generation-2", db_path))
            with jobstore.connect(db_path) as db:
                active = db.execute("SELECT active_generation_id,status FROM jobs WHERE id='doc'").fetchone()
                states = {row["generation_id"]: row["status"] for row in db.execute(
                    "SELECT generation_id,status FROM generations")}
            self.assertEqual((active["active_generation_id"], active["status"]), ("generation-2", "indexed"))
            self.assertEqual(states, {"generation-1": "orphaned", "generation-2": "active"})

    def test_expired_lease_cannot_be_renewed_or_prepare_a_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "jobs.sqlite3"
            with jobstore.connect(db_path) as db:
                db.execute("INSERT INTO jobs(id,sha256,status,attempt_id,locked_at) "
                           "VALUES(?,?, 'processing', ?, '2000-01-01 00:00:00')",
                           ("expired", "expired-hash", "old-attempt"))
                db.commit()
            self.assertFalse(jobstore.heartbeat_job("expired", "old-attempt", db_path))
            self.assertFalse(jobstore.prepare_generation("expired", "old-attempt", "late-generation", 1, db_path))
