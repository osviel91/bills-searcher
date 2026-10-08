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

    def test_worker_indexes_markdown_embedding_and_marks_blank_pdf_failed(self):
        root = Path(self.temp.name)
        with patch.object(worker, "DATA", root), patch.object(worker, "DB", root / "worker.sqlite3"):
            with closing(worker.connect()) as db:
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("good", "hash1", "good.pdf", "documents/good.pdf", "processing"))
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", ("blank", "hash2", "blank.pdf", "documents/blank.pdf", "processing"))
                db.commit()
            worker_es = Mock()
            good_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="Maintenance of printers")))
            blank_markdown = Mock(convert_local=Mock(return_value=SimpleNamespace(markdown="  ")))
            with patch.object(worker, "client", return_value=worker_es), patch.object(worker, "initialize_index"), patch.object(worker, "embed", return_value=[0.5] * 1024) as embed_mock, patch.object(worker, "get_object", return_value=SimpleNamespace(readall=lambda: b"pdf")):
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
            embed_mock.assert_called_once_with("Maintenance of printers")

if __name__ == "__main__":
    unittest.main()
