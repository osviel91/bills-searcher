import json
import os
import subprocess
import sys
import threading
import time
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import httpx
import psycopg
from fastapi import HTTPException
from unittest.mock import patch

import app
import jobstore
from jobstore import claim_next_job, connect, recover_stale_jobs, visible_generation_query
from object_store import delete as delete_object
from storage import INDEX, PASSAGES_INDEX, client


HOOK_DIR = Path(os.environ.get("PHASE0C_HOOK_DIR", "/integration-hooks"))
APP_URL = os.getenv("APP_URL", "http://app:8000")
TOKEN = os.getenv("API_TOKEN", "integration-only-token")


def synthetic_pdf(text):
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    pdf.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(pdf)


@unittest.skipUnless(os.getenv("PHASE0C_INTEGRATION") == "1", "run scripts/run_integration.sh")
class Phase0CIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and len(list(HOOK_DIR.glob("worker-ready-*"))) < 2:
            time.sleep(0.05)
        if len(list(HOOK_DIR.glob("worker-ready-*"))) < 2:
            raise RuntimeError("Both Compose worker processes did not start")

    def setUp(self):
        (HOOK_DIR / "pause-claiming").touch()
        self.doc_ids = []
        self.es = client()
        self.http = httpx.Client(base_url=APP_URL, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
        self._clear_hook()

    def tearDown(self):
        (HOOK_DIR / "pause-claiming").touch()
        self._clear_hook()
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
                try:
                    delete_object(row["object_key"])
                except Exception:
                    pass
            self.es.delete_by_query(index=INDEX, query={"term": {"invoice_id": doc_id}}, refresh=True)
            self.es.delete_by_query(index=PASSAGES_INDEX, query={"term": {"invoice_id": doc_id}}, refresh=True)
        self.http.close()

    def _clear_hook(self):
        for name in ("hook.json", "hook-claimed.json", "hook-reached", "hook-release", "claim-start"):
            (HOOK_DIR / name).unlink(missing_ok=True)

    def _upload(self, body_text):
        filename = f"synthetic-{uuid.uuid4().hex}.pdf"
        response = self.http.post("/api/documents", files={
            "files": (filename, synthetic_pdf(body_text), "application/pdf"),
        })
        self.assertEqual(response.status_code, 200, response.text)
        document = response.json()["documents"][0]
        self.assertEqual(document["status"], "queued", document)
        self.doc_ids.append(document["id"])
        return document["id"]

    def _job(self, doc_id):
        with closing(connect()) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (doc_id,)).fetchone()
        return dict(row)

    def _wait_status(self, doc_id, status, timeout=45):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self._job(doc_id)
            if last.get("status") == status:
                return last
            time.sleep(0.1)
        self.fail(f"Job {doc_id} did not reach {status}; last state was {last}")

    def _run_worker_process(self, job, control=None):
        self._clear_hook()
        if control:
            (HOOK_DIR / "hook.json").write_text(json.dumps({"job_id": job["id"],
                "attempt_id": job["attempt_id"], **control}))
        environment = {**os.environ, "PHASE0C_JOB": json.dumps(job, default=str),
                       "PHASE0C_HOOK_DIR": str(HOOK_DIR)}
        return subprocess.Popen([sys.executable, "tests/integration/worker_runner.py"], env=environment)

    def _wait_hook(self, process, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (HOOK_DIR / "hook-reached").exists():
                return json.loads((HOOK_DIR / "hook-reached").read_text())
            if process.poll() is not None:
                self.fail(f"Worker exited before hook with status {process.returncode}")
            time.sleep(0.02)
        self.fail("Timed out waiting for integration worker hook")

    def _release_hook(self):
        (HOOK_DIR / "hook-release").touch()

    def _publish_initial(self, doc_id):
        job = claim_next_job()
        self.assertEqual(job["id"], doc_id)
        process = self._run_worker_process(job)
        self.assertEqual(process.wait(timeout=45), 0)
        state = self._job(doc_id)
        self.assertEqual(state["status"], "indexed", state["error"])
        return state["active_generation_id"]

    def _requeue_claim(self, doc_id):
        with closing(connect()) as db:
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=?", (doc_id,))
            db.commit()
        job = claim_next_job()
        self.assertEqual(job["id"], doc_id)
        return job

    def test_00_api_to_two_workers_to_search_and_evidence(self):
        doc_id = self._upload(f"Invoice maintenance printer service code {uuid.uuid4().hex}")
        (HOOK_DIR / "pause-claiming").unlink(missing_ok=True)
        try:
            self._wait_status(doc_id, "indexed")
        finally:
            (HOOK_DIR / "pause-claiming").touch()

        state = self._job(doc_id)
        generation = state["active_generation_id"]
        self.assertTrue(generation)
        self.assertEqual(self.es.count(index=INDEX, query={"term": {"generation": generation}})["count"], 1)
        self.assertGreater(self.es.count(index=PASSAGES_INDEX,
                                         query={"term": {"generation": generation}})["count"], 0)
        search = self.http.post("/api/search", json={"query": "maintenance printer service", "limit": 10})
        self.assertEqual(search.status_code, 200, search.text)
        hits = [item for item in search.json()["results"] if item["invoice_id"] == doc_id]
        self.assertEqual(len(hits), 1)
        evidence = self.http.post("/api/search/evidence", json={
            "query": "maintenance printer service", "limit": 10, "invoice_id": doc_id,
        })
        self.assertEqual(evidence.status_code, 200, evidence.text)
        passages = evidence.json()["evidence"]
        self.assertTrue(passages)
        self.assertTrue(all(item["invoice_id"] == doc_id for item in passages))
        self.assertEqual(self.http.get(f"/api/documents/{doc_id}/text").status_code, 200)

        legacy_doc = {"invoice_id": doc_id, "filename": "legacy.pdf", "content": "legacy decoy only",
                      "embedding": [0.1] * 16}
        legacy_passage = {"invoice_id": doc_id, "filename": "legacy.pdf", "generation": "legacy-generation",
                          "active": True, "chunk_index": 0, "start_offset": 0, "end_offset": 17,
                          "content": "legacy decoy only", "embedding": [0.1] * 16}
        self.es.index(index=INDEX, id=doc_id, document=legacy_doc)
        self.es.index(index=PASSAGES_INDEX, id=f"{doc_id}:legacy-generation:0", document=legacy_passage)
        self.es.indices.refresh(index=[INDEX, PASSAGES_INDEX])
        legacy_doc_hits = self.es.search(index=INDEX, size=10, query={"bool": {
            "filter": [visible_generation_query()],
            "must": [{"match_phrase": {"content": "legacy decoy only"}}],
        }})["hits"]["hits"]
        legacy_passage_hits = self.es.search(index=PASSAGES_INDEX, size=10, query={"bool": {
            "filter": [visible_generation_query(passages=True)],
            "must": [{"match_phrase": {"content": "legacy decoy only"}}],
        }})["hits"]["hits"]
        self.assertFalse(legacy_doc_hits)
        self.assertFalse(legacy_passage_hits)
        with patch.object(psycopg, "connect", side_effect=psycopg.OperationalError("injected outage")):
            with self.assertRaises(HTTPException) as error:
                app.run_search("maintenance", 5)
        self.assertEqual(error.exception.status_code, 503)
        with patch.object(psycopg, "connect", side_effect=psycopg.OperationalError("injected outage")):
            with self.assertRaises(HTTPException) as error:
                app.run_invoice_evidence_search("maintenance", 5)
        self.assertEqual(error.exception.status_code, 503)

    def test_two_postgres_connections_claim_one_job_once(self):
        doc_id = self._upload(f"Claim race {uuid.uuid4().hex}")
        start = HOOK_DIR / "claim-start"
        start.unlink(missing_ok=True)
        environment = {**os.environ, "PHASE0C_HOOK_DIR": str(HOOK_DIR)}
        contenders = [subprocess.Popen([sys.executable, "tests/integration/claim_runner.py"],
                                       env=environment, stdout=subprocess.PIPE, text=True)
                      for _ in range(2)]
        start.touch()
        results = [process.communicate(timeout=15)[0].strip() for process in contenders]
        claims = [value for value in results if value]
        self.assertEqual(len(claims), 1, results)
        self.assertEqual(claims[0].split("|", 1)[0], doc_id)

    def test_expired_attempt_cannot_activate_its_prepared_candidate(self):
        doc_id = self._upload(f"Fenced candidate {uuid.uuid4().hex}")
        stale = claim_next_job()
        self.assertEqual(stale["id"], doc_id)
        stale_generation = uuid.uuid4().hex
        self.assertTrue(jobstore.prepare_generation(doc_id, stale["attempt_id"], stale_generation, 0))
        with closing(connect()) as db:
            db.execute("UPDATE jobs SET locked_at=CURRENT_TIMESTAMP-INTERVAL '60 seconds' WHERE id=?", (doc_id,))
            db.commit()
        recover_stale_jobs()
        current = claim_next_job()
        self.assertEqual(current["id"], doc_id)
        self.assertNotEqual(current["attempt_id"], stale["attempt_id"])
        self.assertFalse(jobstore.heartbeat_job(doc_id, stale["attempt_id"]))
        self.assertFalse(jobstore.prepare_generation(doc_id, stale["attempt_id"], uuid.uuid4().hex, 1))
        self.assertFalse(jobstore.activate_generation(doc_id, stale["attempt_id"], stale_generation))
        state = self._job(doc_id)
        self.assertIsNone(state["active_generation_id"])
        self.assertEqual(state["attempt_id"], current["attempt_id"])
        process = self._run_worker_process(current)
        self.assertEqual(process.wait(timeout=45), 0)
        self.assertEqual(self._job(doc_id)["status"], "indexed")

    def test_lease_loss_during_extract_and_index_cannot_publish_late_attempt(self):
        for stage in ("extraction", "document"):
            doc_id = self._upload(f"Stable archived invoice {stage} {uuid.uuid4().hex}")
            old_generation = self._publish_initial(doc_id)
            with closing(connect()) as db:
                db.execute("UPDATE jobs SET status='queued' WHERE id=?", (doc_id,))
                db.commit()
            stale_job = claim_next_job()
            self.assertEqual(stale_job["id"], doc_id)
            first = self._run_worker_process(stale_job, {"stage": stage, "mode": "pause",
                                                         "stop_heartbeat": True})
            self._wait_hook(first)
            with closing(connect()) as db:
                db.execute("UPDATE jobs SET locked_at=CURRENT_TIMESTAMP-INTERVAL '60 seconds' WHERE id=?",
                           (doc_id,))
                db.commit()
            recover_stale_jobs()
            (HOOK_DIR / "pause-claiming").unlink(missing_ok=True)
            try:
                second = self._wait_status(doc_id, "indexed")
                self.assertNotEqual(second.get("attempt_id"), stale_job["attempt_id"])
                replacement_generation = self._job(doc_id)["active_generation_id"]
                self.assertNotEqual(replacement_generation, old_generation)
                with closing(connect()) as db:
                    old_candidate = db.execute("SELECT generation_id FROM generations WHERE job_id=? AND attempt_id=?",
                                               (doc_id, stale_job["attempt_id"])).fetchone()
                if old_candidate:
                    self.assertFalse(jobstore.activate_generation(doc_id, stale_job["attempt_id"],
                                                                   old_candidate["generation_id"]))
                self._release_hook()
                self.assertEqual(first.wait(timeout=30), 0)
            finally:
                (HOOK_DIR / "pause-claiming").touch()
            final = self._job(doc_id)
            self.assertEqual(final["status"], "indexed")
            self.assertEqual(final["active_generation_id"], replacement_generation)
            query = visible_generation_query(path=None)
            self.assertNotIn(stale_job["attempt_id"], str(query))

    def test_partial_elasticsearch_publication_failures_keep_previous_generation_visible(self):
        for stage, index in (("passage", 0), ("document", None), ("refresh", None),
                             ("before_activation", None)):
            doc_id = self._upload(f"Stable invoice failure {stage} {uuid.uuid4().hex}")
            old_generation = self._publish_initial(doc_id)
            job = self._requeue_claim(doc_id)
            process = self._run_worker_process(job, {"stage": stage, "index": index, "mode": "fail"})
            self.assertEqual(process.wait(timeout=45), 0)
            state = self._job(doc_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["active_generation_id"], old_generation)
            self.assertIn(old_generation, visible_generation_query()["bool"]["should"][0]["terms"]["generation"])
            old_hit = self.es.search(index=INDEX, size=5, query={"bool": {"filter": [
                {"term": {"invoice_id": doc_id}}, visible_generation_query(),
            ]}})["hits"]["hits"]
            self.assertEqual(len(old_hit), 1)
            all_doc_generations = self.es.search(index=INDEX, size=20,
                query={"term": {"invoice_id": doc_id}})["hits"]["hits"]
            visible_doc_hits = self.es.search(index=INDEX, size=20, query={"bool": {"filter": [
                {"term": {"invoice_id": doc_id}}, visible_generation_query(),
            ]}})["hits"]["hits"]
            visible_passage_hits = self.es.search(index=PASSAGES_INDEX, size=20, query={"bool": {"filter": [
                {"term": {"invoice_id": doc_id}}, visible_generation_query(passages=True),
            ]}})["hits"]["hits"]
            self.assertEqual({hit["_source"]["generation"] for hit in visible_doc_hits}, {old_generation})
            self.assertEqual({hit["_source"]["generation"] for hit in visible_passage_hits}, {old_generation})
            self.assertTrue(all(hit["_source"]["generation"] != old_generation
                                for hit in all_doc_generations if hit["_id"] != f"{doc_id}:{old_generation}"))

    def test_database_activation_failure_rolls_back_pointer_and_preserves_old_version(self):
        doc_id = self._upload(f"Activation rollback record {uuid.uuid4().hex}")
        old_generation = self._publish_initial(doc_id)
        job = self._requeue_claim(doc_id)
        with closing(connect()) as db:
            db.execute("""CREATE OR REPLACE FUNCTION phase0c_reject_activation() RETURNS trigger AS $$
                BEGIN
                  IF NEW.id = '%s' AND NEW.active_generation_id IS DISTINCT FROM OLD.active_generation_id THEN
                    RAISE EXCEPTION 'injected activation failure';
                  END IF;
                  RETURN NEW;
                END; $$ LANGUAGE plpgsql""" % doc_id)
            db.execute("CREATE TRIGGER phase0c_reject_activation BEFORE UPDATE ON jobs "
                       "FOR EACH ROW EXECUTE FUNCTION phase0c_reject_activation()")
            db.commit()
        try:
            process = self._run_worker_process(job)
            self.assertEqual(process.wait(timeout=45), 0)
            state = self._job(doc_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["active_generation_id"], old_generation)
            self.assertEqual(self.es.count(index=INDEX,
                query={"term": {"generation": old_generation}})["count"], 1)
        finally:
            with closing(connect()) as db:
                db.execute("DROP TRIGGER IF EXISTS phase0c_reject_activation ON jobs")
                db.execute("DROP FUNCTION IF EXISTS phase0c_reject_activation()")
                db.commit()

    def test_search_during_reindex_only_reads_the_previous_generation(self):
        text = f"Searchable legacy source phrase {uuid.uuid4().hex}"
        doc_id = self._upload(text)
        old_generation = self._publish_initial(doc_id)
        job = self._requeue_claim(doc_id)
        process = self._run_worker_process(job, {"stage": "before_activation", "mode": "pause"})
        self._wait_hook(process)
        try:
            for _ in range(8):
                invoice_results = self.http.post("/api/search", json={"query": text, "limit": 10})
                self.assertEqual(invoice_results.status_code, 200, invoice_results.text)
                matching = [hit for hit in invoice_results.json()["results"] if hit["invoice_id"] == doc_id]
                self.assertEqual(len(matching), 1)
                passages = self.http.post("/api/search/evidence", json={
                    "query": text, "limit": 10, "invoice_id": doc_id,
                })
                self.assertEqual(passages.status_code, 200, passages.text)
                for item in passages.json()["evidence"]:
                    self.assertEqual(item["invoice_id"], doc_id)
                self.assertEqual(self._job(doc_id)["active_generation_id"], old_generation)
                visible_documents = self.es.search(index=INDEX, size=10, query={"bool": {"filter": [
                    {"term": {"invoice_id": doc_id}}, visible_generation_query(),
                ]}})["hits"]["hits"]
                visible_passages = self.es.search(index=PASSAGES_INDEX, size=10, query={"bool": {"filter": [
                    {"term": {"invoice_id": doc_id}}, visible_generation_query(passages=True),
                ]}})["hits"]["hits"]
                self.assertEqual({hit["_source"]["generation"] for hit in visible_documents}, {old_generation})
                self.assertEqual({hit["_source"]["generation"] for hit in visible_passages}, {old_generation})
            self._release_hook()
            self.assertEqual(process.wait(timeout=45), 0)
            new_generation = self._job(doc_id)["active_generation_id"]
            self.assertNotEqual(new_generation, old_generation)
        finally:
            self._release_hook()

    def test_cleanup_is_exclusive_batched_retains_active_and_retries_es_failures(self):
        doc_id = self._upload(f"Cleanup safety {uuid.uuid4().hex}")
        active_generation = self._publish_initial(doc_id)
        stale_generation = uuid.uuid4().hex
        old_job_id = f"orphan-{uuid.uuid4()}"
        with closing(connect()) as db:
            db.execute("INSERT INTO generations(generation_id,job_id,attempt_id,status,created_at) "
                       "VALUES(?,?,?,'orphaned',CURRENT_TIMESTAMP-INTERVAL '2 days')",
                       (stale_generation, old_job_id, "finished-attempt"))
            db.commit()
        self.es.index(index=INDEX, id=f"{old_job_id}:{stale_generation}", document={
            "invoice_id": old_job_id, "generation": stale_generation, "content": "orphan",
            "embedding": [0.1] * 16,
        })
        self.es.index(index=PASSAGES_INDEX, id=f"{old_job_id}:{stale_generation}:0", document={
            "invoice_id": old_job_id, "generation": stale_generation, "active": False,
            "content": "orphan", "embedding": [0.1] * 16,
        })
        self.es.indices.refresh(index=[INDEX, PASSAGES_INDEX])
        reached = threading.Event()
        release = threading.Event()

        class BlockingClient:
            def delete_by_query(inner, **kwargs):
                reached.set()
                release.wait(10)
                return self.es.delete_by_query(**kwargs)

        result = []
        first = threading.Thread(target=lambda: result.append(app_worker_cleanup(BlockingClient())))
        first.start()
        self.assertTrue(reached.wait(10))
        second = app_worker_cleanup(self.es)
        self.assertTrue(second["busy"])
        release.set()
        first.join(timeout=15)
        self.assertFalse(first.is_alive())
        self.assertEqual(result[0]["deleted"], 1)
        self.assertEqual(self.es.count(index=INDEX,
            query={"term": {"generation": stale_generation}})["count"], 0)
        self.assertEqual(self.es.count(index=INDEX,
            query={"term": {"generation": active_generation}})["count"], 1)

        failing_generation = uuid.uuid4().hex
        with closing(connect()) as db:
            db.execute("INSERT INTO generations(generation_id,job_id,attempt_id,status,created_at) "
                       "VALUES(?,?,?,'candidate',CURRENT_TIMESTAMP-INTERVAL '2 days')",
                       (failing_generation, f"orphan-{uuid.uuid4()}", "attempt"))
            db.commit()

        class FailingPassagesClient:
            def __init__(inner):
                inner.calls = 0

            def delete_by_query(inner, **kwargs):
                inner.calls += 1
                if kwargs["index"] == PASSAGES_INDEX:
                    raise RuntimeError("injected cleanup failure")
                return self.es.delete_by_query(**kwargs)

        retry_generation = next(iter([failing_generation]))
        failed = app_worker_cleanup(FailingPassagesClient(), batch_size=1)
        self.assertEqual(failed["errors"], 1)
        with closing(connect()) as db:
            self.assertTrue(db.execute("SELECT 1 FROM generations WHERE generation_id=?",
                                       (retry_generation,)).fetchone())
        retried = app_worker_cleanup(self.es, batch_size=1)
        self.assertEqual(retried["deleted"], 1)
        metrics = jobstore.processing_metrics()["maintenance"]
        self.assertGreaterEqual(metrics["generations_cleaned"], 2)
        self.assertGreaterEqual(metrics["generation_cleanup_errors"], 1)


def app_worker_cleanup(es, batch_size=50):
    from worker import cleanup_generations
    return cleanup_generations(es, batch_size)


if __name__ == "__main__":
    unittest.main()
