import hashlib
import os
import threading
import unittest
import uuid
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

import psycopg

from application.template_detection import TemplateDetection
from domain.structured_extraction import BoundingBox, Evidence, ExtractedField
from infrastructure import jobstore
from infrastructure.specialized_extraction_writer import (
    adapt_template_result, write_specialized_result,
)


class CommitAckLost(Exception):
    """The test server committed but the client did not receive its acknowledgement."""


class ObservedConnection:
    def __init__(self, connection, *, before_job_lock=None, after_job_lock=None,
                 release_job_lock=None, lose_commit_ack=False):
        self.connection = connection
        self.before_job_lock = before_job_lock
        self.after_job_lock = after_job_lock
        self.release_job_lock = release_job_lock
        self.lose_commit_ack = lose_commit_ack

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        result = self.connection.__exit__(exc_type, exc, traceback)
        if exc_type is None and self.lose_commit_ack:
            raise CommitAckLost("commit acknowledgement lost after database commit")
        return result

    def execute(self, query, params=()):
        locks_job = "FROM jobs" in query and "FOR UPDATE" in query
        if locks_job and self.before_job_lock:
            self.before_job_lock.set()
        result = self.connection.execute(query, params)
        if locks_job and self.after_job_lock:
            self.after_job_lock.set()
            if not self.release_job_lock.wait(10):
                raise TimeoutError("test did not release the held job lock")
        return result

    def __getattr__(self, name):
        return getattr(self.connection, name)


@unittest.skipUnless(os.getenv("RUN_POSTGRES_CONTRACT_TESTS") == "1", "requires real PostgreSQL")
class SpecializedWriterRecoveryPostgresTests(unittest.TestCase):
    def setUp(self):
        self.document_id = str(uuid.uuid4())
        self.sha256 = hashlib.sha256(self.document_id.encode()).hexdigest()
        self.old_attempt = "old-" + uuid.uuid4().hex
        self.run_ids = []
        with jobstore.connect() as db:
            db.execute("INSERT INTO jobs(id,sha256,status,attempt_id) VALUES(?,?,'indexed',?)",
                       (self.document_id, self.sha256, self.old_attempt))

    def tearDown(self):
        with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL ROLE extraction_retention_operator")
                for run_id in self.run_ids:
                    conn.execute("SELECT retire_extraction_run(%s)", (run_id,))
                conn.execute("RESET ROLE")
                conn.execute("DELETE FROM jobs WHERE id=%s", (self.document_id,))

    def result(self, *, attempt=None, key=None, amount=Decimal("12.30"), extractor_version="1",
               evidence=()):
        return adapt_template_result(
            TemplateDetection("T02", "router", "1", {}, None), "header", extractor_version,
            (ExtractedField("total_due", amount, evidence=tuple(evidence), status="confirmed"),),
            document_id=self.document_id, document_sha256=self.sha256,
            attempt_id=attempt or self.old_attempt, idempotency_key=key,
            extractor_set_fingerprint=f"set-{extractor_version}",
        )

    def rows_for_key(self, key):
        with jobstore.connect() as db:
            return list(db.execute("SELECT id,attempt_id,status FROM extraction_runs "
                                   "WHERE document_id=? AND idempotency_key=?", (self.document_id, key)))

    def run_writer(self, result, errors, values, *, before_job_lock=None,
                   after_job_lock=None, release_job_lock=None):
        original_connect = jobstore.connect

        def observed_connect(path=None):
            return ObservedConnection(original_connect(path), before_job_lock=before_job_lock,
                after_job_lock=after_job_lock, release_job_lock=release_job_lock)

        def write():
            with patch("infrastructure.specialized_extraction_writer.connect", side_effect=observed_connect):
                try:
                    values.append(write_specialized_result(result))
                except Exception as error:
                    errors.append(error)

        thread = threading.Thread(target=write)
        thread.start()
        return thread

    def test_claim_first_fences_old_writer_without_creating_a_run(self):
        writer_started = threading.Event()
        writer_errors, values = [], []
        result = self.result()
        with psycopg.connect(jobstore.DATABASE_URL) as claim_conn:
            with claim_conn.transaction():
                claim_conn.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE", (self.document_id,))
                new_attempt = "new-" + uuid.uuid4().hex
                claim_conn.execute("UPDATE jobs SET status='processing',attempt_id=%s WHERE id=%s",
                                   (new_attempt, self.document_id))
                thread = self.run_writer(result, writer_errors, values, before_job_lock=writer_started)
                self.assertTrue(writer_started.wait(10), "writer did not reach the job-lock query")
        thread.join(10)
        self.assertFalse(thread.is_alive(), "writer did not finish after claim committed")
        self.assertEqual(values, [])
        self.assertEqual(len(writer_errors), 1)
        self.assertIsInstance(writer_errors[0], LookupError)
        self.assertEqual(self.rows_for_key(result.idempotency_key), [])

    def test_write_first_commits_before_new_attempt_can_take_job_lock(self):
        writer_locked = threading.Event()
        release_writer = threading.Event()
        claim_reached_lock = threading.Event()
        writer_errors, values, claim_errors = [], [], []
        result = self.result()
        self.run_ids.append(result.run_id)
        thread = self.run_writer(result, writer_errors, values, after_job_lock=writer_locked,
                                 release_job_lock=release_writer)
        self.assertTrue(writer_locked.wait(10), "writer did not acquire the job lock")

        def claim_new_attempt():
            try:
                with psycopg.connect(jobstore.DATABASE_URL) as conn:
                    with conn.transaction():
                        claim_reached_lock.set()
                        conn.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE", (self.document_id,))
                        conn.execute("UPDATE jobs SET status='processing',attempt_id=%s WHERE id=%s",
                                     ("new-" + uuid.uuid4().hex, self.document_id))
            except Exception as error:
                claim_errors.append(error)

        claim_thread = threading.Thread(target=claim_new_attempt)
        claim_thread.start()
        self.assertTrue(claim_reached_lock.wait(10), "claim did not attempt its row lock")
        release_writer.set()
        thread.join(10)
        claim_thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(claim_thread.is_alive())
        self.assertEqual(writer_errors, [])
        self.assertEqual(claim_errors, [])
        self.assertEqual(len(values), 1)
        self.assertFalse(values[0].reused)
        self.assertEqual(self.rows_for_key(result.idempotency_key)[0]["attempt_id"], self.old_attempt)

    def test_concurrent_same_key_and_payload_create_exactly_one_aggregate(self):
        barrier = threading.Barrier(2)
        outputs, errors = [], []
        first = self.result()
        second = replace(first, run_id="")

        def submit(result):
            try:
                barrier.wait(10)
                outputs.append(write_specialized_result(result))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(item,)) for item in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(sorted(item.reused for item in outputs), [False, True])
        self.assertEqual(outputs[0].run_id, outputs[1].run_id)
        self.run_ids.append(outputs[0].run_id)
        self.assertEqual(len(self.rows_for_key(first.idempotency_key)), 1)

    def test_concurrent_same_key_different_payload_leaves_one_complete_run(self):
        barrier = threading.Barrier(2)
        outputs, errors = [], []
        first = self.result()
        second = self.result(amount=Decimal("99.00"), key=first.idempotency_key)

        def submit(result):
            try:
                barrier.wait(10)
                outputs.append(write_specialized_result(result))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(item,)) for item in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        rows = self.rows_for_key(first.idempotency_key)
        self.assertEqual(len(rows), 1)
        self.run_ids.append(rows[0]["id"])
        with jobstore.connect() as db:
            fields = list(db.execute("SELECT value_json FROM extraction_fields WHERE run_id=?", (rows[0]["id"],)))
        self.assertEqual(len(fields), 1)

    def test_same_key_with_new_extractor_version_conflicts(self):
        first = self.result()
        created = write_specialized_result(first)
        self.run_ids.append(created.run_id)
        newer = self.result(extractor_version="2", key=first.idempotency_key)
        with self.assertRaisesRegex(ValueError, "different result"):
            write_specialized_result(newer)
        self.assertEqual(len(self.rows_for_key(first.idempotency_key)), 1)

    def test_multipage_evidence_creates_each_referenced_page_and_keeps_references(self):
        first = Evidence(1, "12,30", BoundingBox(1, 2, 3, 4), "block-p1", source_token_ids=("tok-p1",))
        second = Evidence(2, "12,30", BoundingBox(5, 6, 7, 8), "block-p2", source_token_ids=("tok-p2",))
        result = self.result(evidence=(first, second))
        written = write_specialized_result(result)
        self.run_ids.append(written.run_id)
        with jobstore.connect() as db:
            pages = [row["page_number"] for row in db.execute(
                "SELECT page_number FROM extraction_pages WHERE run_id=? ORDER BY page_number",
                (written.run_id,))]
            evidence_pages = [row["page_number"] for row in db.execute(
                "SELECT page_number FROM extraction_evidence WHERE run_id=? ORDER BY page_number",
                (written.run_id,))]
        self.assertEqual(pages, [1, 2])
        self.assertEqual(evidence_pages, [1, 2])

    def test_confirmed_rollback_can_retry_with_same_stable_key(self):
        evidence = Evidence(1, "12,30", BoundingBox(1, 2, 3, 4), "block")
        first = self.result(evidence=(evidence,))

        def fail_foreign_key(db, run_id, field_name, line_number, item):
            db.execute("INSERT INTO extraction_evidence(id,run_id,field_name,page_number,quote) "
                       "VALUES(?,?,?,?,?)", (str(uuid.uuid4()), run_id, "missing-field", 1, "synthetic"))

        self.run_ids.append(first.run_id)
        with self.assertRaises(psycopg.Error):
            with patch("infrastructure.specialized_extraction_writer._insert_evidence",
                       side_effect=fail_foreign_key):
                write_specialized_result(first)
        self.assertEqual(self.rows_for_key(first.idempotency_key), [])
        restarted = replace(first, run_id="")
        retried = write_specialized_result(restarted)
        self.run_ids.append(retried.run_id)
        self.assertFalse(retried.reused)
        self.assertEqual(len(self.rows_for_key(first.idempotency_key)), 1)

    def test_lost_commit_ack_is_indeterminate_until_same_key_recovery(self):
        result = self.result()
        self.run_ids.append(result.run_id)
        original_connect = jobstore.connect

        def lose_ack_connect(path=None):
            return ObservedConnection(original_connect(path), lose_commit_ack=True)

        with self.assertRaises(CommitAckLost):
            with patch("infrastructure.specialized_extraction_writer.connect", side_effect=lose_ack_connect):
                write_specialized_result(result)

        with jobstore.connect() as db:
            committed = db.execute("SELECT status,attempt_id FROM extraction_runs WHERE id=?",
                                   (result.run_id,)).fetchone()
        self.assertEqual((committed["status"], committed["attempt_id"]), ("succeeded", self.old_attempt))

        new_attempt = "new-" + uuid.uuid4().hex
        with jobstore.connect() as db:
            db.execute("UPDATE jobs SET status='indexed',attempt_id=? WHERE id=?",
                       (new_attempt, self.document_id))

        stale = replace(result, run_id="")
        with self.assertRaises(LookupError):
            write_specialized_result(stale)

        # A failed recovery connection is still unknown; it does not prove rollback.
        with patch("infrastructure.specialized_extraction_writer.connect",
                   side_effect=psycopg.OperationalError("recovery connection unavailable")):
            with self.assertRaises(psycopg.OperationalError):
                write_specialized_result(replace(result, attempt_id=new_attempt, run_id=""))

        recovered_request = replace(result, attempt_id=new_attempt, run_id="")
        recovered = write_specialized_result(recovered_request)
        self.assertTrue(recovered.reused)
        self.assertEqual((recovered.run_id, recovered.attempt_id), (result.run_id, self.old_attempt))
        self.assertEqual(len(self.rows_for_key(result.idempotency_key)), 1)


if __name__ == "__main__":
    unittest.main()
