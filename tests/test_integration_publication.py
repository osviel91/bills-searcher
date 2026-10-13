import os
import threading
import unittest
import uuid
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import patch

import jobstore
from jobstore import claim_next_job, connect, visible_generation_query
from storage import EMBEDDING_DIMENSIONS, INDEX, PASSAGES_INDEX, client, initialize_index
import worker


@unittest.skipUnless(os.getenv("RUN_INTEGRATION") == "1", "requires live PostgreSQL and Elasticsearch")
class PublicationIntegrationTests(unittest.TestCase):
    def test_two_workers_claim_once_and_publish_a_verified_generation(self):
        es = client()
        initialize_index(es)
        schema = f"phase0b_{uuid.uuid4().hex}"
        doc_id = f"phase0b-{uuid.uuid4()}"
        with jobstore.psycopg.connect(jobstore.DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'CREATE SCHEMA "{schema}"')
        try:
            with patch.dict(os.environ, {"PGOPTIONS": f"-c search_path={schema}"}):
                jobstore._initialized.discard(jobstore.DATABASE_URL)
                with closing(connect()) as db:
                    db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?, 'queued')",
                               (doc_id, doc_id, "synthetic.pdf", f"integration/{doc_id}.pdf"))
                    db.commit()

                claims = []
                start = threading.Barrier(3)

                def claim():
                    start.wait()
                    claims.append(claim_next_job())

                claim_threads = [threading.Thread(target=claim) for _ in range(2)]
                for thread in claim_threads:
                    thread.start()
                start.wait()
                for thread in claim_threads:
                    thread.join()
                claimed = [job for job in claims if job]
                self.assertEqual(len(claimed), 1)

                fake_markdown = SimpleNamespace(convert_local=lambda _: SimpleNamespace(markdown="Synthetic maintenance invoice"))
                with patch.object(worker, "get_object", return_value=SimpleNamespace(readall=lambda: b"%PDF-1.4 synthetic")), \
                     patch.object(worker, "MarkItDown", return_value=fake_markdown), \
                     patch.object(worker, "settings", return_value={"document_mode": "text", "model": "test-model"}), \
                     patch.object(worker, "embed", return_value=[0.1] * EMBEDDING_DIMENSIONS), \
                     patch.object(worker, "embed_many", return_value=[[0.1] * EMBEDDING_DIMENSIONS]):
                    worker.process(claimed[0])

                with closing(connect()) as db:
                    state = db.execute("SELECT status,error,active_generation_id FROM jobs WHERE id=?", (doc_id,)).fetchone()
                self.assertEqual(state["status"], "indexed", state["error"])
                self.assertTrue(state["active_generation_id"])
                generation = state["active_generation_id"]
                self.assertEqual(es.count(index=INDEX, query={"term": {"generation": generation}})["count"], 1)
                self.assertGreater(es.count(index=PASSAGES_INDEX, query={"term": {"generation": generation}})["count"], 0)
                self.assertEqual(visible_generation_query()["bool"]["should"][0],
                                 {"terms": {"generation": [generation]}})

                es.delete_by_query(index=INDEX, query={"term": {"invoice_id": doc_id}}, refresh=True)
                es.delete_by_query(index=PASSAGES_INDEX, query={"term": {"invoice_id": doc_id}}, refresh=True)
        finally:
            jobstore._initialized.discard(jobstore.DATABASE_URL)
            with jobstore.psycopg.connect(jobstore.DATABASE_URL, autocommit=True) as admin:
                admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
