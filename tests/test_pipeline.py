import json
import tempfile
import stat
import unittest
from io import BytesIO
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import worker
import jobstore
from application import search as search_application
from application import ingestion


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [
            patch.object(app, "DATA", root),
            patch.object(app, "DB", root / "jobs.sqlite3"),
            patch.object(search_application, "DB", root / "jobs.sqlite3"),
            patch.object(jobstore, "DATABASE_URL", ""),
            patch.object(app, "API_TOKEN", "test-secret"),
            patch.object(app, "initialize_index"),
            patch.object(app, "ensure_container"),
            patch.object(app, "put_object"),
            patch.object(app, "delete_object"),
            patch.object(search_application, "embed", return_value=[0.25] * 1024),
            patch.object(app, "visible_generation_query", return_value={"terms": {"generation": ["test-generation"]}}),
            patch.object(search_application, "visible_generation_query", return_value={"terms": {"generation": ["test-generation"]}}),
        ]
        for item in self.patches:
            item.start()
        self.es = Mock()
        self.es.search.return_value = {"hits": {"hits": [{
            "_id": "invoice-1",
            "_source": {"invoice_id": "invoice-1", "filename": "invoice.pdf"},
            "_score": 0.82,
        }]}}
        self.es_patch = patch.object(search_application, "es", return_value=self.es)
        self.es_patch.start()
        self.app_es_patch = patch.object(app, "es", return_value=self.es)
        self.app_es_patch.start()
        self.client = TestClient(app.app)
        self.auth = {"Authorization": "Bearer test-secret"}

    def tearDown(self):
        self.es_patch.stop()
        self.app_es_patch.stop()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_multi_upload_normalizes_wrappers_and_deduplicates(self):
        first = b"--boundary\r\nContent-Type: multipart/form-data\r\n\r\n%PDF-1.4\nfixture-one"
        second = b"%PDF-1.4\nfixture-two"
        response = self.client.post("/api/documents", files=[
            ("files", ("one.pdf", first, "application/pdf")),
            ("files", ("one-copy.pdf", first, "application/pdf")),
            ("files", ("two.pdf", second, "application/pdf")),
        ], headers=self.auth)

        self.assertEqual(response.status_code, 200, response.text)
        states = [item["status"] for item in response.json()["documents"]]
        self.assertEqual(states, ["queued", "duplicate", "queued"])
        self.assertEqual(app.put_object.call_count, 2)

    def test_rejects_non_pdf_and_oversized_upload(self):
        bad = self.client.post("/api/documents", files={"files": ("bad.pdf", b"not pdf", "application/pdf")}, headers=self.auth)
        self.assertEqual(bad.status_code, 200)
        self.assertEqual(bad.json()["documents"][0]["status"], "rejected")

        with patch.object(app, "MAX_BYTES", 4):
            large = self.client.post("/api/documents", files={"files": ("large.pdf", b"%PDF-1.4", "application/pdf")}, headers=self.auth)
        self.assertEqual(large.json()["documents"][0]["status"], "rejected")

    def test_image_upload_requires_direct_mode_and_queues_valid_png(self):
        image = BytesIO()
        Image.new("RGB", (2, 2), "red").save(image, format="PNG")
        payload = image.getvalue()
        with patch.object(app, "embedding_settings", return_value={"provider": "jina-local", "document_mode": "direct"}):
            accepted = self.client.post("/api/documents", files={"files": ("invoice.png", payload, "image/png")}, headers=self.auth)
        self.assertEqual(accepted.json()["documents"][0]["status"], "queued")
        self.assertTrue(app.put_object.call_args.args[0].endswith(".png"))
        self.assertEqual(app.put_object.call_args.args[2], "image/png")

        with patch.object(app, "embedding_settings", return_value={"provider": "jina-local", "document_mode": "text"}):
            rejected = self.client.post("/api/documents", files={"files": ("invoice.png", payload, "image/png")}, headers=self.auth)
        self.assertEqual(rejected.json()["documents"][0]["status"], "rejected")

    def test_search_generates_query_vector_and_returns_document(self):
        response = self.client.post("/api/search", json={"query": "reparación de impresoras", "limit": 5}, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["results"][0]["filename"], "invoice.pdf")
        search_application.embed.assert_called_once_with("reparación de impresoras", query=True)
        self.assertEqual(self.es.search.call_count, 2)
        lexical, semantic = [call.kwargs for call in self.es.search.call_args_list]
        self.assertIn("content", str(lexical["query"]))
        self.assertIn("knn", semantic)

    def test_search_falls_back_to_bm25_when_embeddings_fail(self):
        self.es.search.return_value = {"hits": {"hits": [{
            "_id": "invoice-1", "_source": {"invoice_id": "invoice-1", "filename": "invoice.pdf"},
        }]}}
        with patch.object(search_application, "embed", side_effect=RuntimeError("provider unavailable")):
            result = app.run_search("repair", 5)
        self.assertEqual(result["results"][0]["invoice_id"], "invoice-1")
        self.assertEqual(self.es.search.call_count, 1)

    def test_knn_candidate_count_respects_search_window_at_max_limit(self):
        self.es.search.return_value = {"hits": {"hits": []}}
        app.run_search("repair", 50)
        knn = self.es.search.call_args_list[1].kwargs["knn"]
        self.assertEqual(knn["k"], 500)
        self.assertGreaterEqual(knn["num_candidates"], knn["k"])

    def test_evidence_search_falls_back_to_bm25_when_embeddings_fail(self):
        self.es.search.return_value = {"hits": {"hits": [{"_id": "passage-1", "_source": {
            "invoice_id": "invoice-1", "filename": "invoice.pdf", "content": "repair", "chunk_index": 0,
        }}]}}
        with patch.object(search_application, "embed", side_effect=RuntimeError("provider unavailable")):
            result = app.run_invoice_evidence_search("repair")
        self.assertEqual(result["evidence"][0]["passage_id"], "passage-1")
        self.assertEqual(self.es.search.call_count, 1)

    def test_worker_rejects_a_stale_attempt_before_publication(self):
        root = Path(self.temp.name)
        with patch.object(worker, "DATA", root), patch.object(worker, "DB", root / "fenced.sqlite3"):
            with closing(worker.connect()) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status,attempt_id) VALUES(?,?,?,?,?,?)",
                           ("fenced", "fenced-hash", "fenced.pdf", "documents/fenced.pdf", "processing", "new"))
                db.commit()
            with self.assertRaisesRegex(RuntimeError, "no longer current"):
                worker._ensure_current_attempt({"id": "fenced", "attempt_id": "old"})

    def test_startup_does_not_hide_elasticsearch_initialization_failure(self):
        with patch.object(app, "initialize_index", side_effect=RuntimeError("mapping unavailable")), \
             patch.object(app, "container_client") as container, \
             patch.object(app, "embedding_health", return_value=(False, False)):
            container.return_value.get_container_properties.return_value = None
            with TestClient(app.app) as client:
                response = client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["elasticsearch"])
        self.assertEqual(response.json()["index_initialization_error"], "RuntimeError")

    def test_direct_invoice_filter_uses_structured_fields_and_returns_total(self):
        self.es.search.return_value = {"hits": {"total": {"value": 1}, "hits": [{
            "_source": {"invoice_id": "invoice-1", "customer_code": "001234"},
        }]}}
        result = app.run_invoice_filter(customer_code="001234", concept="wallet comida",
                                        invoice_date_from="2026-01-01", total_payable_min=10,
                                        limit=20, offset=40)

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["file_url"], "/api/documents/invoice-1/file")
        request = self.es.search.call_args.kwargs
        self.assertEqual(request["from_"], 40)
        self.assertIn("customer_code", str(request["query"]))
        self.assertIn("concepts", str(request["query"]))

    def test_evidence_search_returns_passage_with_traceable_offsets(self):
        hit = {"_id": "invoice-1:generation:0", "_score": 0.7, "_source": {
            "invoice_id": "invoice-1", "filename": "invoice.pdf", "chunk_index": 0,
            "start_offset": 120, "end_offset": 154, "content": "Concepto: mantenimiento de equipos",
        }}
        self.es.search.return_value = {"hits": {"hits": [hit]}}

        result = app.run_invoice_evidence_search("mantenimiento", invoice_id="invoice-1")

        self.assertEqual(result["evidence"][0]["content"], "Concepto: mantenimiento de equipos")
        self.assertEqual(result["evidence"][0]["start_offset"], 120)
        self.assertEqual(self.es.search.call_count, 2)
        self.assertIn("generation", str(self.es.search.call_args_list[1].kwargs["knn"]["filter"]))

    def test_evidence_rest_endpoint_is_authenticated_and_validates_query(self):
        with patch.object(app, "run_invoice_evidence_search", return_value={"evidence": []}) as search:
            response = self.client.post("/api/search/evidence", json={"query": "mantenimiento", "limit": 3}, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        search.assert_called_once_with("mantenimiento", 3, None)

        invalid = self.client.post("/api/search/evidence", json={"query": "  "}, headers=self.auth)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.client.post("/api/search/evidence", json={"query": "texto"}).status_code, 401)

    def test_filter_rest_endpoint_validates_ranges_and_returns_page(self):
        self.es.search.return_value = {"hits": {"total": {"value": 1}, "hits": [{
            "_source": {"invoice_id": "invoice-1", "filename": "invoice.pdf", "customer_code": "001234"},
        }]}}
        response = self.client.post("/api/invoices/filter", json={
            "customer_code": "001234", "invoice_date_from": "2026-01-01", "limit": 20,
        }, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["results"][0]["file_url"], "/api/documents/invoice-1/file")
        self.assertEqual(self.client.post("/api/invoices/filter", json={"customer_code": "001234"}).status_code, 401)

        invalid = self.client.post("/api/invoices/filter", json={
            "invoice_date_from": "2026-12-31", "invoice_date_to": "2026-01-01",
        }, headers=self.auth)
        self.assertEqual(invalid.status_code, 422)

    def test_evaluation_cases_persist_locally_and_run_recall(self):
        payload = {"cases": [{
            "id": "case-1", "query": "mantenimiento de impresoras", "relevant_evidence": [],
        }]}
        loaded = self.client.get("/api/evaluation/evidence/cases", headers=self.auth)
        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertEqual(loaded.json(), {"cases": []})

        saved = self.client.put("/api/evaluation/evidence/cases", json=payload, headers=self.auth)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["saved"], 1)
        self.assertEqual(stat.S_IMODE((Path(self.temp.name) / "evidence-evaluation.json").stat().st_mode), 0o600)
        self.assertEqual(self.client.get("/api/evaluation/evidence/cases", headers=self.auth).json()["cases"],
                         payload["cases"])
        self.assertEqual(self.client.get("/api/evaluation/evidence/cases").status_code, 401)
        self.assertEqual(self.client.post("/api/evaluation/evidence/run", json=payload,
                                          headers=self.auth).status_code, 422)

        run_payload = {"cases": [{**payload["cases"][0], "relevant_evidence": [
            {"invoice_id": "invoice-1", "quote": "maintenance of printers"},
        ]}]}
        with patch.object(app, "run_search", return_value={"results": [{"invoice_id": "invoice-1"}]}) as invoice_search, \
             patch.object(app, "run_invoice_evidence_search", return_value={"evidence": [{
                "invoice_id": "invoice-1", "content": "Maintenance of printers service", "score": 0.4,
                "filename": "invoice.pdf", "chunk_index": 0, "start_offset": 0, "end_offset": 32,
                "file_url": "/api/documents/invoice-1/file",
             }]}) as search:
            result = self.client.post("/api/evaluation/evidence/run", json=run_payload, headers=self.auth)
        self.assertEqual(result.status_code, 200, result.text)
        evaluated = result.json()["results"][0]
        self.assertEqual(evaluated["invoice_recall_at_5"], 1.0)
        self.assertEqual(evaluated["invoice_recall_at_10"], 1.0)
        self.assertEqual(evaluated["global_evidence_recall_at_10"], 1.0)
        self.assertEqual(evaluated["scoped_evidence_recall_at_10"], 1.0)
        self.assertEqual(result.json()["summary"]["mean_scoped_evidence_recall_at_10"], 1.0)
        invoice_search.assert_called_once_with("mantenimiento de impresoras", limit=10)
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[0].kwargs, {"limit": 10})
        self.assertEqual(search.call_args_list[1].kwargs, {"limit": 10, "invoice_id": "invoice-1"})

    def test_extracted_text_endpoint_requires_auth_and_returns_source_text(self):
        with closing(app.connect()) as db:
            db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status,active_generation_id) VALUES(?,?,?,?,?,?)",
                       ("invoice-1", "text-hash", "invoice.pdf", "documents/invoice.pdf", "indexed", "test-generation"))
            db.commit()
        self.es.search.return_value = {"hits": {"hits": [{"_source": {
            "invoice_id": "invoice-1", "filename": "invoice.pdf", "content": "Synthetic extracted quote",
        }}]}}
        response = self.client.get("/api/documents/invoice-1/text", headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["content"], "Synthetic extracted quote")
        self.assertEqual(self.client.get("/api/documents/invoice-1/text").status_code, 401)

    def test_reindex_invoices_requeues_existing_indexed_documents(self):
        with closing(app.connect()) as db:
            for doc_id, status in (("old-1", "indexed"), ("old-2", "indexed"), ("queued", "queued")):
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)",
                           (doc_id, doc_id, f"{doc_id}.pdf", f"documents/{doc_id}.pdf", status))
            db.commit()

        result = app.reindex_invoices()

        self.assertEqual(result["requeued"], 2)
        with closing(app.connect()) as db:
            statuses = {row["id"]: row["status"] for row in db.execute("SELECT id,status FROM jobs")}
        self.assertEqual(statuses, {"old-1": "queued", "old-2": "queued", "queued": "queued"})

    def test_protected_endpoints_require_token_and_documents_are_paginated(self):
        unauthorized = self.client.get("/api/documents")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(self.client.post("/mcp", json={}).status_code, 401)
        response = self.client.post("/api/documents", files=[
            ("files", ("one.pdf", b"%PDF-1.4\nfixture", "application/pdf")),
            ("files", ("bad.txt", b"not pdf", "text/plain")),
        ], headers=self.auth)
        self.assertEqual(response.status_code, 200)
        page = self.client.get("/api/documents?limit=1&offset=0", headers=self.auth)
        self.assertEqual(page.json()["total"], 1)
        self.assertEqual(len(page.json()["documents"]), 1)

    def test_mcp_ingestion_tool_queues_pdf(self):
        import base64

        with patch.object(app, "put_object"):
            result = app.index_invoice("agent.pdf", base64.b64encode(b"%PDF-1.4\nagent invoice").decode())
        self.assertEqual(result["status"], "queued")
        self.assertEqual(app.get_invoice_status(result["id"])["status"], "queued")

    def test_mcp_streamable_http_connects_lists_tools_and_calls_search_evidence_and_status(self):
        headers = {**self.auth, "Accept": "application/json, text/event-stream"}
        initialized = self.client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1"}},
        })
        self.assertEqual(initialized.status_code, 200, initialized.text)
        self.assertIn("protocolVersion", initialized.json()["result"])
        self.assertEqual(self.client.post("/mcp", headers={**headers,
            "MCP-Protocol-Version": initialized.json()["result"]["protocolVersion"]}, json={
                "jsonrpc": "2.0", "method": "notifications/initialized",
            }).status_code, 202)
        headers["MCP-Protocol-Version"] = initialized.json()["result"]["protocolVersion"]

        def rpc(request_id, method, params=None):
            response = self.client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {},
            })
            self.assertEqual(response.status_code, 200, response.text)
            return response.json()["result"]

        tools = rpc(2, "tools/list")["tools"]
        names = {tool["name"] for tool in tools}
        self.assertTrue({"semantic_search", "search_invoice_evidence", "get_invoice_status"} <= names)

        with patch.object(app, "run_search", return_value={"results": [{"invoice_id": "invoice-1"}]}) as search, \
             patch.object(app, "run_invoice_evidence_search", return_value={"evidence": [{"invoice_id": "invoice-1", "content": "synthetic quote"}]}) as evidence:
            self.assertIn("invoice-1", json.dumps(rpc(3, "tools/call", {
                "name": "semantic_search", "arguments": {"query": "maintenance"},
            })))
            self.assertIn("synthetic quote", json.dumps(rpc(4, "tools/call", {
                "name": "search_invoice_evidence", "arguments": {"query": "maintenance"},
            })))
        search.assert_called_once_with("maintenance", 10)
        evidence.assert_called_once_with("maintenance", 5, None)

        status = rpc(5, "tools/call", {
            "name": "get_invoice_status", "arguments": {"document_id": "missing-invoice"},
        })
        self.assertIn("Documento no encontrado", json.dumps(status))

    def test_home_shows_upload_feedback(self):
        page = self.client.get("/").text
        self.assertIn("Enviando", page)
        self.assertIn("No se pudo subir", page)
        self.assertIn("Buscar evidencia", page)
        self.assertIn("Evaluación", page)
        self.assertIn("Añadir cita a este caso", page)
        self.assertIn("/api/search/evidence", page)
        self.assertIn("/api/invoices/filter", page)
        self.assertIn("/api/evaluation/evidence/cases", page)
        self.assertIn("/api/evaluation/evidence/run", page)

    def test_swagger_docs_and_openapi_security_are_configured(self):
        docs = self.client.get("/api/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("/api/openapi.json", docs.text)
        self.assertEqual(self.client.get("/api/redoc").status_code, 200)

        schema = self.client.get("/api/openapi.json").json()
        self.assertEqual(schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"], "bearer")
        self.assertEqual(schema["paths"]["/api/search"]["post"]["security"], [{"HTTPBearer": []}])
        self.assertNotIn("security", schema["paths"]["/api/health"]["get"])
        self.assertIn("UploadResponse", schema["components"]["schemas"])
        self.assertIn("422", schema["paths"]["/api/search"]["post"]["responses"])

    def test_worker_does_not_mark_job_indexed_when_passages_remain_inactive(self):
        root = Path(self.temp.name)
        with patch.object(worker, "DATA", root), patch.object(worker, "DB", root / "worker-publication.sqlite3"):
            with closing(worker.connect()) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status,attempt_id) VALUES(?,?,?,?,?,?)",
                           ("publication", "hash-publication", "publication.pdf", "documents/publication.pdf", "processing", "attempt-publication"))
                db.commit()
            worker_es = Mock()
            worker_es.count.return_value = {"count": 0}
            markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="Synthetic invoice text")))
            with patch.object(ingestion, "client", return_value=worker_es), patch.object(ingestion, "settings", return_value={
                    "provider": "openai-compatible", "document_mode": "text", "model": "test-model"}), \
                 patch.object(ingestion, "embed", return_value=[0.5] * 1024), \
                 patch.object(ingestion, "embed_many", return_value=[[0.5] * 1024]), \
                 patch.object(ingestion, "get_object", return_value=SimpleNamespace(readall=lambda: b"synthetic pdf")), \
                 patch.object(ingestion, "MarkItDown", return_value=markdown):
                worker.process({"id": "publication", "attempt_id": "attempt-publication", "sha256": "hash-publication", "filename": "publication.pdf",
                                "object_key": "documents/publication.pdf"})

            with closing(worker.connect()) as db:
                job = db.execute("SELECT status,error FROM jobs WHERE id='publication'").fetchone()
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "RuntimeError")

    def test_worker_indexes_markdown_embedding_and_marks_blank_pdf_failed(self):
        root = Path(self.temp.name)
        with patch.object(worker, "DATA", root), patch.object(worker, "DB", root / "worker.sqlite3"):
            with closing(worker.connect()) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("good", "hash1", "good.pdf", "documents/good.pdf", "queued"))
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("blank", "hash2", "blank.pdf", "documents/blank.pdf", "queued"))
                db.commit()
            claimed = {jobstore.claim_next_job(worker.DB)["id"]: None for _ in range(2)}
            with closing(worker.connect()) as db:
                claimed = {row["id"]: dict(row) for row in db.execute(
                    "SELECT * FROM jobs WHERE status='processing'")}
            good_job, blank_job = claimed["good"], claimed["blank"]
            worker_es = Mock()
            worker_es.count.return_value = {"count": 1}
            good_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="Maintenance of printers")))
            blank_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="  ")))
            with patch.object(ingestion, "client", return_value=worker_es), patch.object(ingestion, "settings", return_value={"provider": "openai-compatible", "document_mode": "text", "model": "test-model"}), patch.object(ingestion, "embed", return_value=[0.5] * 1024) as embed_mock, patch.object(ingestion, "embed_many", return_value=[[0.5] * 1024]) as embed_many_mock, patch.object(ingestion, "get_object", return_value=SimpleNamespace(readall=lambda: b"pdf")):
                with patch.object(ingestion, "MarkItDown", return_value=good_markdown):
                    with patch("application.structured_extraction.extract_document",
                               side_effect=RuntimeError("structured store unavailable")):
                        worker.process(good_job)
                with patch.object(ingestion, "MarkItDown", return_value=blank_markdown):
                    worker.process(blank_job)

            with closing(worker.connect()) as db:
                statuses = dict(db.execute("SELECT id,status FROM jobs").fetchall())
                error = db.execute("SELECT error FROM jobs WHERE id='blank'").fetchone()[0]
            self.assertEqual(statuses, {"good": "indexed", "blank": "failed"})
            self.assertEqual(error, "ValueError")
            indexed_document = worker_es.index.call_args.kwargs["document"]
            self.assertEqual(indexed_document["content"], "Maintenance of printers")
            self.assertEqual(indexed_document["concepts"], [])
            self.assertEqual(indexed_document["confidence"], {})
            embed_mock.assert_called_once_with("Maintenance of printers")
            embed_many_mock.assert_called_once_with(["Maintenance of printers"])
            passage_document = worker_es.index.call_args_list[0].kwargs["document"]
            self.assertFalse(passage_document["active"])
            self.assertEqual(passage_document["content"], "Maintenance of printers")
            self.assertEqual(worker_es.indices.refresh.call_count, 1)

if __name__ == "__main__":
    unittest.main()
