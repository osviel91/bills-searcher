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
            with closing(connect()) as db:
                db.execute("DELETE FROM extraction_evidence WHERE run_id IN "
                           "(SELECT id FROM extraction_runs WHERE document_id=?)", (doc_id,))
                for table in ("extraction_fields", "extraction_lines", "extraction_pages"):
                    db.execute(f"DELETE FROM {table} WHERE run_id IN "
                               "(SELECT id FROM extraction_runs WHERE document_id=?)", (doc_id,))
                db.execute("DELETE FROM extraction_runs WHERE document_id=?", (doc_id,))
                row = db.execute("SELECT object_key FROM jobs WHERE id=?", (doc_id,)).fetchone()
                db.execute("DELETE FROM generations WHERE job_id=?", (doc_id,))
                db.execute("DELETE FROM jobs WHERE id=?", (doc_id,))
                db.commit()
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

    def _run_job(self, data, *, enabled, mode=None, obsolete=False):
        doc_id = self._upload(data)
        job = claim_next_job()
        self.assertEqual(job["id"], doc_id)
        diagnostic_path = HOOK_DIR / f"diagnostic-{doc_id}.json"
        environment = {**os.environ, "PHASE0C_JOB": json.dumps(job, default=str),
                       "PHASE0C_HOOK_DIR": str(HOOK_DIR),
                       "AS400_SHADOW_ENABLED": "true" if enabled else "false",
                       "SHADOW_EXPECT_ENABLED": "true" if enabled else "false"}
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

    def _assert_generic_and_published(self, doc_id):
        with closing(connect()) as db:
            job = db.execute("SELECT status,active_generation_id FROM jobs WHERE id=?", (doc_id,)).fetchone()
            runs = list(db.execute("SELECT extractor_version,status FROM extraction_runs WHERE document_id=?",
                                   (doc_id,)))
            fields = list(db.execute("SELECT name FROM extraction_fields WHERE run_id IN "
                                     "(SELECT id FROM extraction_runs WHERE document_id=?)", (doc_id,)))
            generations = list(db.execute("SELECT generation_id,status FROM generations WHERE job_id=?",
                                          (doc_id,)))
        self.assertEqual(job["status"], "indexed", dict(job))
        self.assertTrue(job["active_generation_id"])
        self.assertEqual([(row["extractor_version"], row["status"]) for row in runs],
                         [("generic-invoice@1", "succeeded")])
        self.assertTrue(fields)
        self.assertEqual([(row["generation_id"], row["status"]) for row in generations],
                         [(job["active_generation_id"], "active")])
        docs = self.es.search(index=INDEX, size=5, query={"term": {"invoice_id": doc_id}})["hits"]["hits"]
        passages = self.es.search(index=PASSAGES_INDEX, size=20,
                                  query={"term": {"invoice_id": doc_id}})["hits"]["hits"]
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

    def test_templates_unknown_and_disabled_vs_enabled_timings(self):
        timings = []
        expectations = {"T01": ("as400-t01-header-summary@1", "as400-t01-lines@1"),
                        "T02": ("as400-t02-header-summary@1", "as400-t02-lines@1"),
                        "unknown": ()}
        for template, specialized_versions in expectations.items():
            data = fixture(template)
            self.assertEqual(detect_template(extract_pdf(data)).template, template)
            for enabled in (False, True):
                doc_id, diagnostic, total_ms = self._run_job(data, enabled=enabled)
                generation = self._assert_generic_and_published(doc_id)
                if not enabled:
                    self.assertIsNone(diagnostic)
                    shadow_ms = 0
                else:
                    self.assertIsNotNone(diagnostic)
                    self.assertEqual(diagnostic["template"], template, diagnostic)
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

    def test_shadow_child_failure_and_timeout_are_isolated(self):
        timings = []
        for mode, expected_status in (("failure", "worker_error"), ("timeout", "timeout")):
            data = fixture("T01")
            doc_id, diagnostic, total_ms = self._run_job(data, enabled=True, mode=mode)
            self._assert_generic_and_published(doc_id)
            self.assertEqual(diagnostic["status"], expected_status)
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
        doc_id, diagnostic, _ = self._run_job(data, enabled=True, mode="stale", obsolete=True)
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
