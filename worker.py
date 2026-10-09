import os
import time
import tempfile
import uuid
from contextlib import closing
from pathlib import Path

from markitdown import MarkItDown

import mimetypes

from embeddings import embed, embed_file, embed_many, settings
from invoice_metadata import extract_invoice_metadata
from jobstore import claim_next_job, connect as connect_jobs, recover_stale_jobs
from object_store import ensure_container, get as get_object
from passages import chunk_text
from storage import INDEX, PASSAGES_INDEX, client, initialize_index

DATA = Path(os.getenv("DATA_DIR", "/data"))
DB = DATA / "jobs.sqlite3"


def connect():
    return connect_jobs(DB)


def process(job):
    try:
        config = settings()
        content_type = mimetypes.guess_type(job["filename"].lower())[0] or "application/octet-stream"
        data = get_object(job["object_key"]).readall()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf:
            if content_type == "application/pdf":
                pdf.write(data)
                pdf.flush()
                try:
                    markdown = MarkItDown(enable_plugins=False).convert_local(pdf.name).markdown.strip()
                except Exception:
                    if config["document_mode"] != "direct":
                        raise
                    markdown = ""
            else:
                markdown = ""
        if not markdown and config["document_mode"] != "direct":
            raise ValueError("MarkItDown no encontró texto; OCR está desactivado. Activa Jina multimodal directo para PDFs escaneados")
        metadata = extract_invoice_metadata(markdown)
        passages = chunk_text(markdown)
        passage_vectors = []
        for offset in range(0, len(passages), 32):
            passage_vectors.extend(embed_many([item["text"] for item in passages[offset:offset + 32]]))
        vector = (embed_file(data, content_type, job["id"]) if config["document_mode"] == "direct"
                  else embed(markdown))
        es = client()
        generation = uuid.uuid4().hex
        for index, (passage, passage_vector) in enumerate(zip(passages, passage_vectors)):
            es.index(index=PASSAGES_INDEX, id=f"{job['id']}:{generation}:{index}", document={
                "invoice_id": job["id"], "filename": job["filename"],
                "generation": generation, "active": False, "chunk_index": index,
                "start_offset": passage["start_offset"], "end_offset": passage["end_offset"],
                "content": passage["text"], "embedding": passage_vector,
            })
        es.index(index=INDEX, id=job["id"], document={
            "invoice_id": job["id"], "sha256": job["sha256"],
            "filename": job["filename"], "model": settings()["model"],
            "content": markdown, **metadata, "embedding": vector,
        })
        published = es.update_by_query(
            index=PASSAGES_INDEX,
            query={"term": {"invoice_id": job["id"]}},
            script={"source": "ctx._source.active = (ctx._source.generation == params.generation)",
                    "params": {"generation": generation}},
            refresh=True,
        )
        if published.get("failures"):
            raise RuntimeError("No se pudieron publicar todos los pasajes de la factura")
        active_passages = es.count(index=PASSAGES_INDEX, query={"bool": {"filter": [
            {"term": {"invoice_id": job["id"]}}, {"term": {"generation": generation}},
            {"term": {"active": True}},
        ]}})["count"]
        if active_passages != len(passages):
            raise RuntimeError("No se pudieron publicar todos los pasajes activos de la factura")
        try:
            es.delete_by_query(index=PASSAGES_INDEX, refresh=True,
                               query={"bool": {"filter": [{"term": {"invoice_id": job["id"]}},
                                                            {"term": {"active": False}}]}})
        except Exception:
            pass
        status, error = "indexed", None
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    with closing(connect()) as db:
        db.execute("UPDATE jobs SET status=?,error=?,locked_at=NULL WHERE id=?", (status, error, job["id"]))
        db.commit()


def main():
    ensure_container()
    es = client()
    initialize_index(es)
    recover_stale_jobs(DB)
    while True:
        job = claim_next_job(DB)
        if job:
            process(job)
        else:
            time.sleep(2)


if __name__ == "__main__":
    main()
