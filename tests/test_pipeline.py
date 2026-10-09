import tempfile
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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [
            patch.object(app, "DATA", root),
            patch.object(app, "DB", root / "jobs.sqlite3"),
            patch.object(jobstore, "DATABASE_URL", ""),
            patch.object(app, "API_TOKEN", "test-secret"),
            patch.object(app, "initialize_index"),
            patch.object(app, "ensure_container"),
            patch.object(app, "put_object"),
            patch.object(app, "delete_object"),
            patch.object(app, "embed", return_value=[0.25] * 1024),
        ]
        for item in self.patches:
            item.start()
        self.es = Mock()
        self.es.search.return_value = {"hits": {"hits": [{
            "_id": "invoice-1",
            "_source": {"invoice_id": "invoice-1", "filename": "invoice.pdf"},
            "_score": 0.82,
        }]}}
        self.es_patch = patch.object(app, "es", return_value=self.es)
        self.es_patch.start()
        self.client = TestClient(app.app)
        self.auth = {"Authorization": "Bearer test-secret"}

    def tearDown(self):
        self.es_patch.stop()
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
        app.embed.assert_called_once_with("reparación de impresoras", query=True)
        self.assertEqual(self.es.search.call_count, 2)
        lexical, semantic = [call.kwargs for call in self.es.search.call_args_list]
        self.assertIn("content", str(lexical["query"]))
        self.assertIn("knn", semantic)

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
        self.assertIn("active", str(self.es.search.call_args_list[1].kwargs["knn"]["filter"]))

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

    def test_home_shows_upload_feedback(self):
        page = self.client.get("/").text
        self.assertIn("Enviando", page)
        self.assertIn("No se pudo subir", page)
        self.assertIn("Buscar evidencia", page)
        self.assertIn("/api/search/evidence", page)
        self.assertIn("/api/invoices/filter", page)

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
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)",
                           ("publication", "hash-publication", "publication.pdf", "documents/publication.pdf", "processing"))
                db.commit()
            worker_es = Mock()
            worker_es.update_by_query.return_value = {"failures": [], "updated": 1}
            worker_es.count.return_value = {"count": 0}
            markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="Synthetic invoice text")))
            with patch.object(worker, "client", return_value=worker_es), patch.object(worker, "settings", return_value={
                    "provider": "openai-compatible", "document_mode": "text", "model": "test-model"}), \
                 patch.object(worker, "embed", return_value=[0.5] * 1024), \
                 patch.object(worker, "embed_many", return_value=[[0.5] * 1024]), \
                 patch.object(worker, "get_object", return_value=SimpleNamespace(readall=lambda: b"synthetic pdf")), \
                 patch.object(worker, "MarkItDown", return_value=markdown):
                worker.process({"id": "publication", "sha256": "hash-publication", "filename": "publication.pdf",
                                "object_key": "documents/publication.pdf"})

            with closing(worker.connect()) as db:
                job = db.execute("SELECT status,error FROM jobs WHERE id='publication'").fetchone()
            self.assertEqual(job["status"], "failed")
            self.assertIn("pasajes activos", job["error"])

    def test_worker_indexes_markdown_embedding_and_marks_blank_pdf_failed(self):
        root = Path(self.temp.name)
        with patch.object(worker, "DATA", root), patch.object(worker, "DB", root / "worker.sqlite3"):
            with closing(worker.connect()) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("good", "hash1", "good.pdf", "documents/good.pdf", "processing"))
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("blank", "hash2", "blank.pdf", "documents/blank.pdf", "processing"))
                db.commit()
            worker_es = Mock()
            worker_es.update_by_query.return_value = {"failures": []}
            worker_es.count.return_value = {"count": 1}
            good_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="Maintenance of printers")))
            blank_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="  ")))
            with patch.object(worker, "client", return_value=worker_es), patch.object(worker, "initialize_index"), patch.object(worker, "settings", return_value={"provider": "openai-compatible", "document_mode": "text", "model": "test-model"}), patch.object(worker, "embed", return_value=[0.5] * 1024) as embed_mock, patch.object(worker, "embed_many", return_value=[[0.5] * 1024]) as embed_many_mock, patch.object(worker, "get_object", return_value=SimpleNamespace(readall=lambda: b"pdf")):
                with patch.object(worker, "MarkItDown", return_value=good_markdown):
                    worker.process({"id": "good", "sha256": "hash1", "filename": "good.pdf", "object_key": "documents/good.pdf"})
                with patch.object(worker, "MarkItDown", return_value=blank_markdown):
                    worker.process({"id": "blank", "sha256": "hash2", "filename": "blank.pdf", "object_key": "documents/blank.pdf"})

            with closing(worker.connect()) as db:
                statuses = dict(db.execute("SELECT id,status FROM jobs").fetchall())
                error = db.execute("SELECT error FROM jobs WHERE id='blank'").fetchone()[0]
            self.assertEqual(statuses, {"good": "indexed", "blank": "failed"})
            self.assertIn("OCR está desactivado", error)
            indexed_document = worker_es.index.call_args.kwargs["document"]
            self.assertEqual(indexed_document["content"], "Maintenance of printers")
            self.assertEqual(indexed_document["concepts"], [])
            self.assertEqual(indexed_document["confidence"], {})
            embed_mock.assert_called_once_with("Maintenance of printers")
            embed_many_mock.assert_called_once_with(["Maintenance of printers"])
            passage_document = worker_es.index.call_args_list[0].kwargs["document"]
            self.assertFalse(passage_document["active"])
            self.assertEqual(passage_document["content"], "Maintenance of printers")
            worker_es.update_by_query.assert_called_once()

if __name__ == "__main__":
    unittest.main()
