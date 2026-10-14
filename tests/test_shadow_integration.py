import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import httpx
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from application.template_detection import detect_template
from infrastructure.jobstore import claim_next_job, connect
from infrastructure.object_store import container_client, delete as delete_object
from infrastructure.pdf_extraction import extract_pdf
from infrastructure.storage import INDEX, PASSAGES_INDEX, client

HOOK_DIR = Path(os.environ.get("PHASE0C_HOOK_DIR", "/integration-hooks"))
APP_URL = os.getenv("APP_URL", "http://app:8000")
TOKEN = os.getenv("API_TOKEN", "integration-only-token")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "integration"))
from shadow_fixtures import fixture


@unittest.skipUnless(os.getenv("PHASE0C_INTEGRATION") == "1", "run scripts/run_shadow_integration.sh")
class ShadowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv("AS400_SHADOW_ENABLED") != "true":
            raise RuntimeError("Shadow integration service is not configured enabled")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and len(list(HOOK_DIR.glob("worker-ready-*"))) < 2:
            time.sleep(0.05)
        if len(list(HOOK_DIR.glob("worker-ready-*"))) < 2:
            raise RuntimeError("Both Compose worker processes did not start")

    def setUp(self):
        (HOOK_DIR / "pause-claiming").touch()
        self.doc_ids = []
        self.blob_baselines = {}
        self.es = client()
        self.http = httpx.Client(base_url=APP_URL,
                                 headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
        self._clear_hooks()

    def tearDown(self):
        (HOOK_DIR / "pause-claiming").touch()
        (HOOK_DIR / "shadow-child-release").touch()
        for doc_id in self.doc_ids:
            with psycopg.connect(os.environ["DATABASE_ADMIN_URL"], row_factory=dict_row) as db:
                with db.transaction():
                    db.execute("SET LOCAL ROLE extraction_retention_operator")
                    run_ids = [row["id"] for row in db.execute(
                        "SELECT id FROM extraction_runs WHERE document_id=%s", (doc_id,))]
                    for run_id in run_ids:
                        db.execute("SELECT retire_extraction_run(%s)", (run_id,))
                    db.execute("RESET ROLE")
                row = db.execute("SELECT object_key FROM jobs WHERE id=%s", (doc_id,)).fetchone()
                db.execute("DELETE FROM generations WHERE job_id=%s", (doc_id,))
                db.execute("DELETE FROM jobs WHERE id=%s", (doc_id,))
            if row and row["object_key"]:
                delete_object(row["object_key"])
            self.es.delete_by_query(index=INDEX, query={"term": {"invoice_id": doc_id}}, refresh=True)
            self.es.delete_by_query(index=PASSAGES_INDEX,
                                    query={"term": {"invoice_id": doc_id}}, refresh=True)
        self.http.close()

    def _clear_hooks(self):
        for path in HOOK_DIR.glob("diagnostic-*.json"):
            path.unlink(missing_ok=True)
        for name in ("shadow-child-started", "shadow-child-release"):
            (HOOK_DIR / name).unlink(missing_ok=True)
        for path in HOOK_DIR.glob("runner-complete-*.json"):
            path.unlink(missing_ok=True)

    def _upload(self, data):
        # The API deduplicates by bytes; a PDF comment preserves content while isolating each run.
        data += f"\n% run-{uuid.uuid4().hex}\n".encode()
        blob_baseline = {
            blob.name: (str(blob.etag), blob.size) for blob in container_client().list_blobs()
        }
        response = self.http.post("/api/documents", files={
            "files": (f"synthetic-{uuid.uuid4().hex}.pdf", data, "application/pdf"),
        })
        self.assertEqual(response.status_code, 200, response.text)
        doc_id = response.json()["documents"][0]["id"]
        self.doc_ids.append(doc_id)
        self.blob_baselines[doc_id] = blob_baseline
        return doc_id

    def _run_job(self, data, *, enabled, persist=False, mode=None, obsolete=False,
                 shadow_timeout=None):
        doc_id = self._upload(data)
        job = claim_next_job()
        self.assertEqual(job["id"], doc_id)
        return self._run_claimed_job(job, enabled=enabled, persist=persist, mode=mode,
                                     obsolete=obsolete, shadow_timeout=shadow_timeout)

    def _run_claimed_job(self, job, *, enabled, persist=False, mode=None, obsolete=False,
                         shadow_timeout=None):
        doc_id = job["id"]
        diagnostic_path = HOOK_DIR / f"diagnostic-{doc_id}.json"
        diagnostic_path.unlink(missing_ok=True)
        (HOOK_DIR / f"runner-complete-{doc_id}.json").unlink(missing_ok=True)
        environment = {**os.environ, "PHASE0C_JOB": json.dumps(job, default=str),
                       "PHASE0C_HOOK_DIR": str(HOOK_DIR),
                       "AS400_SHADOW_ENABLED": "true" if enabled else "false",
                       "SHADOW_EXPECT_ENABLED": "true" if enabled else "false",
                       "AS400_SHADOW_PERSISTENCE_ENABLED": "true" if persist else "false",
                       "SHADOW_EXPECT_PERSISTENCE": "true" if persist else "false"}
        if shadow_timeout is not None:
            environment["SHADOW_TEST_TIMEOUT_SECONDS"] = str(shadow_timeout)
        if mode:
            environment["SHADOW_TEST_MODE"] = mode
        started = time.perf_counter()
        process = subprocess.Popen([sys.executable, "tests/integration/worker_runner.py"],
                                   cwd=ROOT, env=environment)
        if obsolete:
            marker = HOOK_DIR / "shadow-child-started"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not marker.exists() and process.poll() is None:
                time.sleep(0.02)
            if not marker.exists():
                returncode = process.wait(timeout=35)
                with closing(connect()) as db:
                    state = db.execute("SELECT status,error FROM jobs WHERE id=?", (doc_id,)).fetchone()
                self.fail(f"shadow child missed fencing barrier; worker exit={returncode}, job={dict(state)}")
            with closing(connect()) as db:
                state = db.execute("SELECT status,active_generation_id FROM jobs WHERE id=?",
                                   (doc_id,)).fetchone()
                self.assertEqual(state["status"], "indexed")
                db.execute("UPDATE jobs SET attempt_id=? WHERE id=?", ("replacement-attempt", doc_id))
                db.commit()
            (HOOK_DIR / "shadow-child-release").touch()
            generation_before = state["active_generation_id"]
        self.assertEqual(process.wait(timeout=35), 0)
        total_ms = round((time.perf_counter() - started) * 1000)
        self.assertEqual(json.loads((HOOK_DIR / f"runner-complete-{doc_id}.json").read_text()),
                         {"active_children": 0})
        if obsolete:
            with closing(connect()) as db:
                state_after = db.execute("SELECT status,attempt_id,active_generation_id FROM jobs WHERE id=?",
                                         (doc_id,)).fetchone()
            self.assertEqual(state_after["status"], "indexed")
            self.assertEqual(state_after["attempt_id"], "replacement-attempt")
            self.assertEqual(state_after["active_generation_id"], generation_before)
        diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
        return doc_id, diagnostic, total_ms

    def _assert_generic_and_published(self, doc_id, *, specialized=False, template=None,
                                      specialized_status="succeeded"):
        with closing(connect()) as db:
            job = db.execute("SELECT status,active_generation_id FROM jobs WHERE id=?", (doc_id,)).fetchone()
            runs = list(db.execute("SELECT id,extractor_version,status,execution_mode,template,"
                                   "abstention_reason,attempt_id FROM extraction_runs WHERE document_id=?",
                                   (doc_id,)))
            fields = list(db.execute("SELECT name FROM extraction_fields WHERE run_id IN "
                                     "(SELECT id FROM extraction_runs WHERE document_id=?)", (doc_id,)))
            generations = list(db.execute("SELECT generation_id,status FROM generations WHERE job_id=?",
                                          (doc_id,)))
        self.assertEqual(job["status"], "indexed", dict(job))
        self.assertTrue(job["active_generation_id"])
        generic_runs = [row for row in runs if row["extractor_version"] == "generic-invoice@1"]
        specialized_runs = [row for row in runs if row["execution_mode"] == "shadow-specialized"]
        self.assertTrue(generic_runs)
        self.assertTrue(all((row["extractor_version"], row["status"]) ==
                            ("generic-invoice@1", "succeeded") for row in generic_runs))
        self.assertEqual(len(specialized_runs), int(specialized))
        if specialized:
            self.assertEqual((specialized_runs[0]["template"], specialized_runs[0]["status"]),
                             (template, specialized_status))
            if template == "unknown":
                self.assertTrue(specialized_runs[0]["abstention_reason"])
                with closing(connect()) as db:
                    count = db.execute("SELECT count(*) AS n FROM extraction_fields WHERE run_id=?",
                                       (specialized_runs[0]["id"],)).fetchone()["n"]
                self.assertEqual(count, 0)
        self.assertTrue(fields)
        self.assertEqual([(row["generation_id"], row["status"]) for row in generations
                          if row["status"] == "active"],
                         [(job["active_generation_id"], "active")])
        active_query = {"bool": {"filter": [
            {"term": {"invoice_id": doc_id}},
            {"term": {"generation": job["active_generation_id"]}},
        ]}}
        docs = self.es.search(index=INDEX, size=5, query=active_query)["hits"]["hits"]
        passages = self.es.search(index=PASSAGES_INDEX, size=20,
                                  query=active_query)["hits"]["hits"]
        self.assertEqual(len(docs), 1)
        self.assertTrue(passages)
        self.assertEqual({hit["_source"]["generation"] for hit in docs + passages},
                         {job["active_generation_id"]})
        with closing(connect()) as db:
            object_key = db.execute("SELECT object_key FROM jobs WHERE id=?", (doc_id,)).fetchone()["object_key"]
        self.assertTrue(container_client().get_blob_client(object_key).exists())
        blobs_after = {blob.name: (str(blob.etag), blob.size)
                       for blob in container_client().list_blobs()}
        expected_blobs = dict(self.blob_baselines[doc_id])
        expected_blobs[object_key] = blobs_after[object_key]
        self.assertEqual(blobs_after, expected_blobs)
        return job["active_generation_id"]

    def _assert_specialized_evidence(self, doc_id, template):
        expected_line_extractor = f"as400-{template.lower()}-lines"
        with closing(connect()) as db:
            run = db.execute("SELECT id,template,status FROM extraction_runs WHERE document_id=? "
                             "AND execution_mode='shadow-specialized'", (doc_id,)).fetchone()
            lines = list(db.execute("SELECT line_number,description FROM extraction_lines WHERE run_id=?",
                                    (run["id"],)))
            line_evidence = list(db.execute("SELECT line_number,page_number,quote,bounds_json,method,"
                                            "extractor_id,extractor_version,source_token_ids_json "
                                            "FROM extraction_evidence WHERE run_id=? AND line_number IS NOT NULL",
                                            (run["id"],)))
        self.assertEqual((run["template"], run["status"]), (template, "succeeded"))
        self.assertTrue(lines)
        self.assertTrue(line_evidence)
        self.assertTrue(all(item["line_number"] in {line["line_number"] for line in lines}
                            and item["page_number"] == 1
                            and item["quote"] and json.loads(item["bounds_json"])
                            and item["method"]
                            and item["extractor_id"] == expected_line_extractor
                            and item["extractor_version"] == "1"
                            and json.loads(item["source_token_ids_json"])
                            for item in line_evidence))
        for line in lines:
            description_words = set((line["description"] or "").upper().split())
            evidence_quotes = [item["quote"].upper() for item in line_evidence
                               if item["line_number"] == line["line_number"]]
            self.assertTrue(description_words)
            self.assertTrue(all(any(word in quote for quote in evidence_quotes)
                                for word in description_words))

        if template == "T02":
            with closing(connect()) as db:
                field = db.execute("SELECT observation_status FROM extraction_fields WHERE run_id=? "
                                   "AND name='tax_rates'", (run["id"],)).fetchone()
                evidence = db.execute("SELECT quote,bounds_json,method,"
                                      "extractor_id,extractor_version,source_token_ids_json "
                                      "FROM extraction_evidence WHERE run_id=? AND field_name='tax_rates'",
                                      (run["id"],)).fetchone()
            self.assertEqual(field["observation_status"], "confirmed")
            self.assertTrue(evidence["quote"] and json.loads(evidence["bounds_json"]))
            self.assertEqual((evidence["method"], evidence["extractor_id"],
                              evidence["extractor_version"]),
                             ("detail-vat-column", "as400-t02-header-summary", "1"))
            self.assertTrue(json.loads(evidence["source_token_ids_json"]))

    def test_templates_unknown_and_disabled_vs_enabled_timings(self):
        timings = []
        expectations = {"T01": ("as400-t01-header-summary@1", "as400-t01-lines@1"),
                        "T02": ("as400-t02-header-summary@1", "as400-t02-lines@1"),
                        "unknown": ()}
        for template, specialized_versions in expectations.items():
            data = fixture(template)
            self.assertEqual(detect_template(extract_pdf(data)).template, template)
            for enabled in (False, True):
                doc_id, diagnostic, total_ms = self._run_job(
                    data, enabled=enabled, shadow_timeout=10 if enabled else None)
                generation = self._assert_generic_and_published(doc_id)
                if not enabled:
                    self.assertIsNone(diagnostic)
                    shadow_ms = 0
                else:
                    self.assertIsNotNone(diagnostic)
                    self.assertEqual(diagnostic["template"], template, diagnostic)
                    self.assertEqual(diagnostic["persistence_status"], "disabled")
                    if template == "unknown":
                        self.assertEqual(diagnostic["status"], "abstained")
                        self.assertEqual(diagnostic["extractor_versions"], ["generic-invoice@1"])
                    else:
                        self.assertEqual(diagnostic["status"], "succeeded")
                        self.assertTrue(set(specialized_versions).issubset(diagnostic["extractor_versions"]))
                    encoded = json.dumps(diagnostic)
                    for private_marker in ("synthetic-private-financial-text", "12,00", "10,00",
                                           "SERVICIO SINTETICO", "ARTICULO SINTETICO", "COMISION",
                                           "BASE IMPONIBLE", "DOCUMENTO SINTETICO"):
                        self.assertNotIn(private_marker, encoded)
                    shadow_ms = diagnostic["duration_ms"]
                timings.append({"template": template, "shadow_enabled": enabled,
                                "total_ms": total_ms, "shadow_ms": shadow_ms,
                                "active_generation": generation})
        for template in expectations:
            baseline = next(item for item in timings
                            if item["template"] == template and not item["shadow_enabled"])
            enabled = next(item for item in timings
                           if item["template"] == template and item["shadow_enabled"])
            enabled["observed_overhead_ms"] = enabled["total_ms"] - baseline["total_ms"]
        print("SHADOW_SYNTHETIC_TIMINGS " + json.dumps(timings, sort_keys=True))

    def test_persistence_enabled_stores_t01_t02_and_unknown_after_indexing(self):
        for template in ("T01", "T02", "unknown"):
            with self.subTest(template=template):
                doc_id, diagnostic, _ = self._run_job(fixture(template), enabled=True, persist=True,
                                                      shadow_timeout=10)
                self.assertEqual(diagnostic["persistence_enabled"], True)
                self.assertEqual(diagnostic["persistence_status"], "created", diagnostic)
                self.assertEqual(diagnostic["shadow_status"],
                                 "abstained" if template == "unknown" else "succeeded")
                self.assertTrue(diagnostic["idempotency_key"].startswith("specialized-v1:"))
                self.assertEqual(diagnostic["persistence_attempt_id"], diagnostic["attempt_id"])
                self._assert_generic_and_published(doc_id, specialized=True, template=template)
                if template in {"T01", "T02"}:
                    self._assert_specialized_evidence(doc_id, template)
                encoded = json.dumps(diagnostic)
                for private_marker in ("synthetic-private-financial-text", "12,00", "10,00",
                                       "SERVICIO SINTETICO", "ARTICULO SINTETICO", "COMISION",
                                       "BASE IMPONIBLE", "DOCUMENTO SINTETICO"):
                    self.assertNotIn(private_marker, encoded)

    def test_reindex_reuses_persisted_run_without_attempt_reassignment(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T02"), enabled=True, persist=True,
                                              shadow_timeout=10)
        original_attempt = diagnostic["attempt_id"]
        original_generation = self._assert_generic_and_published(doc_id, specialized=True, template="T02")
        with closing(connect()) as db:
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=? AND status='indexed'", (doc_id,))
            db.commit()
        job = claim_next_job()
        self.assertEqual(job["id"], doc_id)
        _, replay, _ = self._run_claimed_job(job, enabled=True, persist=True, shadow_timeout=10)
        self.assertEqual(replay["persistence_status"], "reused")
        self.assertEqual(replay["persistence_attempt_id"], original_attempt)
        with closing(connect()) as db:
            runs = list(db.execute("SELECT attempt_id FROM extraction_runs WHERE document_id=? "
                                   "AND execution_mode='shadow-specialized'", (doc_id,)))
            state = db.execute("SELECT status,active_generation_id FROM jobs WHERE id=?", (doc_id,)).fetchone()
        self.assertEqual([row["attempt_id"] for row in runs], [original_attempt])
        self.assertEqual(state["status"], "indexed")
        self.assertNotEqual(state["active_generation_id"], original_generation)
        self._assert_generic_and_published(doc_id, specialized=True, template="T02")

    def test_commit_ack_loss_keeps_key_and_reindex_recovers_original_run(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=True, persist=True,
                                              mode="commit-ack-lost", shadow_timeout=10)
        self.assertEqual(diagnostic["persistence_status"], "indeterminate")
        stable_key = diagnostic["idempotency_key"]
        original_attempt = diagnostic["attempt_id"]
        self._assert_generic_and_published(doc_id, specialized=True, template="T01")
        with closing(connect()) as db:
            before = list(db.execute("SELECT id,attempt_id FROM extraction_runs WHERE document_id=? "
                                     "AND execution_mode='shadow-specialized'", (doc_id,)))
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=? AND status='indexed'", (doc_id,))
            db.commit()
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["attempt_id"], original_attempt)
        current = claim_next_job()
        self.assertEqual(current["id"], doc_id)
        _, replay, _ = self._run_claimed_job(current, enabled=True, persist=True, shadow_timeout=10)
        self.assertEqual(replay["persistence_status"], "reused")
        self.assertEqual(replay["idempotency_key"], stable_key)
        self.assertEqual(replay["persistence_attempt_id"], original_attempt)
        with closing(connect()) as db:
            after = list(db.execute("SELECT id,attempt_id FROM extraction_runs WHERE document_id=? "
                                    "AND execution_mode='shadow-specialized'", (doc_id,)))
        self.assertEqual(after, before)
        self._assert_generic_and_published(doc_id, specialized=True, template="T01")

    def test_adapter_failure_keeps_document_searchable_and_writes_no_partial_run(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=True, persist=True,
                                              mode="adapter-failure")
        self.assertEqual(diagnostic["persistence_status"], "adapter_error")
        self._assert_generic_and_published(doc_id)
        search = self.http.post("/api/search", json={"query": "SERVICIO SINTETICO", "limit": 5})
        self.assertEqual(search.status_code, 200, search.text)

    def test_extractor_error_persists_failed_run_without_partial_observations(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=True, persist=True,
                                              mode="extractor-error")
        self.assertEqual((diagnostic["shadow_status"], diagnostic["persistence_status"]),
                         ("extractor_error", "created"))
        self._assert_generic_and_published(doc_id, specialized=True, template="T01",
                                           specialized_status="failed")
        with closing(connect()) as db:
            shadow_run = db.execute("SELECT id FROM extraction_runs WHERE document_id=? "
                                    "AND execution_mode='shadow-specialized'", (doc_id,)).fetchone()
            counts = [db.execute(f"SELECT count(*) AS n FROM {table} WHERE run_id=?", (shadow_run["id"],))
                      .fetchone()["n"] for table in ("extraction_fields", "extraction_lines", "extraction_evidence")]
        self.assertEqual(counts, [0, 0, 0])
        self.assertNotIn("synthetic private extractor failure", json.dumps(diagnostic))

    def test_sql_persistence_failure_isolated_from_index_and_search(self):
        suffix = uuid.uuid4().hex
        trigger_name, function_name = f"b2c_fail_{suffix}", f"b2c_fail_fn_{suffix}"
        with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as admin:
            admin.execute(sql.SQL("""CREATE FUNCTION public.{}() RETURNS trigger LANGUAGE plpgsql AS $body$
                BEGIN
                    IF NEW.execution_mode='shadow-specialized' THEN
                        RAISE EXCEPTION 'synthetic persistence constraint failure' USING ERRCODE='23503';
                    END IF;
                    RETURN NEW;
                END $body$""").format(sql.Identifier(function_name)))
            admin.execute(sql.SQL("CREATE TRIGGER {} BEFORE INSERT ON public.extraction_runs "
                                  "FOR EACH ROW EXECUTE FUNCTION public.{}()")
                          .format(sql.Identifier(trigger_name), sql.Identifier(function_name)))
        try:
            doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=True, persist=True)
            self.assertEqual(diagnostic["persistence_status"], "error")
            self.assertEqual(diagnostic["persistence_error_code"], "transaction-rolled-back")
            self._assert_generic_and_published(doc_id)
            search = self.http.post("/api/search", json={"query": "SERVICIO SINTETICO", "limit": 5})
            self.assertEqual(search.status_code, 200, search.text)
        finally:
            with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as admin:
                admin.execute(sql.SQL("DROP TRIGGER IF EXISTS {} ON public.extraction_runs")
                              .format(sql.Identifier(trigger_name)))
                admin.execute(sql.SQL("DROP FUNCTION IF EXISTS public.{}()")
                              .format(sql.Identifier(function_name)))

    def test_shadow_extractor_timeout_never_persists_partial_observations(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=True, persist=True, mode="timeout")
        self.assertEqual(diagnostic["status"], "timeout")
        self.assertEqual(diagnostic["persistence_status"], "not_attempted")
        self._assert_generic_and_published(doc_id)

    def test_persistence_enabled_is_still_inert_when_shadow_is_disabled(self):
        doc_id, diagnostic, _ = self._run_job(fixture("T01"), enabled=False, persist=True)
        self.assertIsNone(diagnostic)
        self._assert_generic_and_published(doc_id)

    def test_shadow_child_failure_and_timeout_are_isolated(self):
        timings = []
        for mode, expected_status in (("failure", "worker_error"), ("timeout", "timeout")):
            data = fixture("T01")
            doc_id, diagnostic, total_ms = self._run_job(data, enabled=True, persist=True, mode=mode)
            self._assert_generic_and_published(doc_id)
            self.assertEqual(diagnostic["status"], expected_status)
            self.assertEqual(diagnostic["persistence_status"], "not_attempted")
            self.assertNotIn("synthetic-private-financial-text", json.dumps(diagnostic))
            if mode == "timeout":
                self.assertLess(total_ms, 20_000)
                self.assertLessEqual(diagnostic["duration_ms"], 10_000)
            self.assertEqual(json.loads((HOOK_DIR / f"runner-complete-{doc_id}.json").read_text()),
                             {"active_children": 0})
            timings.append({"scenario": mode, "total_ms": total_ms,
                            "shadow_ms": diagnostic["duration_ms"], "status": diagnostic["status"]})
        print("SHADOW_FAULT_TIMINGS " + json.dumps(timings, sort_keys=True))

    def test_obsolete_attempt_is_discarded_after_real_pipeline_publication(self):
        data = fixture("T01")
        doc_id, diagnostic, _ = self._run_job(data, enabled=True, persist=True,
                                              mode="stale", obsolete=True)
        self.assertIsNone(diagnostic)
        generation = self._assert_generic_and_published(doc_id)
        with closing(connect()) as db:
            current = db.execute("SELECT attempt_id,active_generation_id,status FROM jobs WHERE id=?",
                                 (doc_id,)).fetchone()
        self.assertEqual(current["attempt_id"], "replacement-attempt")
        self.assertEqual(current["active_generation_id"], generation)
        self.assertEqual(current["status"], "indexed")


if __name__ == "__main__":
    unittest.main()
