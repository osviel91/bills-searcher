import os
import time
import tempfile
from contextlib import closing
from pathlib import Path

from markitdown import MarkItDown

import mimetypes

from embeddings import embed, embed_file, settings
from jobstore import claim_next_job, connect as connect_jobs, recover_stale_jobs
from object_store import ensure_container, get as get_object
from storage import INDEX, client, initialize_index

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
        vector = (embed_file(data, content_type, job["id"]) if config["document_mode"] == "direct"
                  else embed(markdown))
        es = client()
        es.index(index=INDEX, id=job["id"], document={
            "invoice_id": job["id"], "sha256": job["sha256"],
            "filename": job["filename"], "model": settings()["model"],
            "content": markdown, "embedding": vector,
        })
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
