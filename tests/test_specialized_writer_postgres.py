import os
import hashlib
import unittest
import uuid
from decimal import Decimal
from unittest.mock import patch

import psycopg

from domain.structured_extraction import BoundingBox, Evidence, ExtractedField, ProposedLine
from infrastructure import jobstore
from infrastructure.specialized_extraction_writer import Component, SpecializedResult, write_specialized_result


@unittest.skipUnless(os.getenv("RUN_POSTGRES_CONTRACT_TESTS") == "1", "requires real PostgreSQL")
class SpecializedWriterPostgresTests(unittest.TestCase):
    def setUp(self):
        self.document_id = str(uuid.uuid4())
        self.sha = hashlib.sha256(self.document_id.encode()).hexdigest()
        self.attempt = "attempt-" + uuid.uuid4().hex
        self.keys = []
        with jobstore.connect() as db:
            db.execute("INSERT INTO jobs(id,sha256,status,attempt_id) VALUES(?,?,'indexed',?)",
                       (self.document_id, self.sha, self.attempt))

    def tearDown(self):
        with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL ROLE extraction_retention_operator")
                for run_id in self.keys:
                    conn.execute("SELECT retire_extraction_run(%s)", (run_id,))
                conn.execute("RESET ROLE")
                conn.execute("DELETE FROM jobs WHERE id=%s", (self.document_id,))

    def result(self, *, key=None, attempt=None, fields=None, lines=(), template="T02",
               components=None, fingerprint="fingerprint-v1"):
        return SpecializedResult(
            self.document_id, self.sha, attempt or self.attempt, "specialized", "router", "1", template,
            components or (Component("header", "1"), Component("lines", "2")), fingerprint, 1,
            key or str(uuid.uuid4()), tuple(fields if fields is not None else (
                ExtractedField("total_due", Decimal("12.30"), status="confirmed"),)), tuple(lines), "succeeded",
        )

    def test_atomic_fields_lines_multiple_evidence_and_idempotent_reuse_keep_original_attempt(self):
        key = str(uuid.uuid4())
        evidence = Evidence(1, "12,30", BoundingBox(1, 2, 3, 4), "block", method="column",
                            extractor_id="lines", extractor_version="2", source_token_ids=("token",))
        fields = (ExtractedField("total_due", Decimal("12.30"), evidence=(evidence, evidence),
                                 status="confirmed"),)
        lines = (ProposedLine("SYNTHETIC", {"amount": Decimal("12.30")}, (evidence,)),)
        result = self.result(key=key, fields=fields, lines=lines)
        first = write_specialized_result(result)
        self.keys.append(first.run_id)
        current_attempt = "new-authority"
        with jobstore.connect() as db:
            db.execute("UPDATE jobs SET attempt_id=? WHERE id=?", (current_attempt, self.document_id))
        replay = write_specialized_result(self.result(key=key, attempt=current_attempt, fields=fields, lines=lines))
        self.assertTrue(replay.reused)
        self.assertEqual((replay.run_id, replay.attempt_id), (first.run_id, self.attempt))
        with jobstore.connect() as db:
            counts = [db.execute(f"SELECT count(*) AS n FROM {table} WHERE {column}=?",
                                 (first.run_id,)).fetchone()["n"]
                      for table, column in (("extraction_fields", "run_id"),
                                            ("extraction_lines", "run_id"),
                                            ("extraction_evidence", "run_id"),
                                            ("extraction_pages", "run_id"))]
        self.assertEqual(counts, [1, 1, 3, 1])

    def test_conflicting_payload_with_same_key_is_rejected(self):
        key = str(uuid.uuid4())
        first = write_specialized_result(self.result(key=key))
        self.keys.append(first.run_id)
        with self.assertRaisesRegex(ValueError, "different result"):
            write_specialized_result(self.result(key=key, fields=(
                ExtractedField("total_due", Decimal("99.00"), status="confirmed"),)))

    def test_stale_attempt_is_rejected_before_idempotent_lookup(self):
        key = str(uuid.uuid4())
        first = write_specialized_result(self.result(key=key))
        self.keys.append(first.run_id)
        with jobstore.connect() as db:
            db.execute("UPDATE jobs SET attempt_id='current-attempt' WHERE id=?", (self.document_id,))
        with self.assertRaisesRegex(LookupError, "no longer authoritative"):
            write_specialized_result(self.result(key=key, attempt=self.attempt))

    def test_invalid_evidence_foreign_key_rolls_back_entire_run(self):
        bad_evidence = Evidence(9, "synthetic", None)
        result = self.result(fields=(ExtractedField("total_due", Decimal("12.30"),
                            evidence=(bad_evidence,), status="confirmed"),))
        self.keys.append(result.run_id)
        def violate_fk(db, run_id, field_name, line_number, evidence):
            db.execute("INSERT INTO extraction_evidence(id,run_id,field_name,page_number,quote) "
                       "VALUES(?,?,?,?,?)", (str(uuid.uuid4()), run_id, "missing-field", 9, "synthetic"))
        with self.assertRaises(psycopg.Error):
            with patch("infrastructure.specialized_extraction_writer._insert_evidence", side_effect=violate_fk):
                write_specialized_result(result)
        with jobstore.connect() as db:
            self.assertIsNone(db.execute("SELECT id FROM extraction_runs WHERE id=?", (result.run_id,)).fetchone())

    def test_terminal_run_is_not_overwritten(self):
        key = str(uuid.uuid4())
        first = write_specialized_result(self.result(key=key))
        self.keys.append(first.run_id)
        with self.assertRaises(psycopg.Error):
            with psycopg.connect(jobstore.DATABASE_URL) as conn:
                conn.execute("UPDATE extraction_runs SET error='changed' WHERE id=%s", (first.run_id,))

    def test_t01_roles_unknown_abstention_and_component_version_are_persisted(self):
        fields = (ExtractedField("displayed_total_euros", Decimal("124.15"),
                                  status="unverified_semantics"),
                  ExtractedField("total_due", None, status="ambiguous"))
        t01 = write_specialized_result(self.result(template="T01", fields=fields))
        self.keys.append(t01.run_id)
        upgraded = write_specialized_result(self.result(
            key=str(uuid.uuid4()), components=(Component("header", "2"),), fingerprint="fingerprint-v2"))
        self.keys.append(upgraded.run_id)
        unknown = self.result(key=str(uuid.uuid4()), template="unknown", fields=(), lines=())
        from dataclasses import replace
        unknown = replace(unknown, abstention_reason="insufficient-structural-signals")
        abstained = write_specialized_result(unknown)
        self.keys.append(abstained.run_id)
        with jobstore.connect() as db:
            roles = {row["name"]: row["observation_status"] for row in db.execute(
                "SELECT name,observation_status FROM extraction_fields WHERE run_id=?", (t01.run_id,))}
            version_row = db.execute("SELECT extractor_version,result_fingerprint_version FROM extraction_runs WHERE id=?",
                                     (upgraded.run_id,)).fetchone()
            unknown_row = db.execute("SELECT status,template,abstention_reason FROM extraction_runs WHERE id=?",
                                     (abstained.run_id,)).fetchone()
        self.assertEqual(roles, {"displayed_total_euros": "unverified_semantics", "total_due": "ambiguous"})
        self.assertIn('"version":"2"', version_row["extractor_version"])
        self.assertEqual(version_row["result_fingerprint_version"], 1)
        self.assertEqual(tuple(unknown_row.values()), ("succeeded", "unknown", "insufficient-structural-signals"))

    def test_runtime_writer_role_has_no_admin_or_retention_privileges(self):
        with jobstore.connect() as db:
            privileges = db.execute("""SELECT r.rolsuper,
                    pg_catalog.pg_has_role(current_user,'extraction_retention_operator','member') AS can_retain,
                    pg_catalog.has_schema_privilege(current_user,'public','CREATE') AS can_create
                FROM pg_roles r WHERE r.rolname=current_user""").fetchone()
        self.assertFalse(privileges["rolsuper"])
        self.assertFalse(privileges["can_retain"])
        self.assertFalse(privileges["can_create"])


if __name__ == "__main__":
    unittest.main()
