import mimetypes
import logging
import os
import tempfile
import threading
import uuid
from contextlib import closing
from pathlib import Path

from markitdown import MarkItDown

from domain.invoice_metadata import extract_invoice_metadata
from domain.passages import chunk_text
from infrastructure.embeddings import embed, embed_file, embed_many, settings
from infrastructure.jobstore import (
    activate_generation,
    connect as connect_jobs,
    delete_generation,
    HEARTBEAT_SECONDS,
    heartbeat_job,
    LEASE_SECONDS,
    lock_generation_cleanup,
    prepare_generation,
    record_cleanup_metrics,
    record_publication_error,
    stale_generations,
)
from infrastructure.object_store import get as get_object
from infrastructure.storage import INDEX, PASSAGES_INDEX, client, initialize_index

DATA = Path(os.getenv("DATA_DIR", "/data"))
DB = DATA / "jobs.sqlite3"


def process(job, stage_hook=None, *, db_path=None):
    db_path = db_path or DB
    stop_heartbeat = threading.Event()
    attempt_id = job.get("attempt_id")
    generation = None
    heartbeat = (threading.Thread(target=lambda: _heartbeat_until_stopped(
        job["id"], attempt_id, stop_heartbeat, db_path), daemon=True) if attempt_id else None)
    if heartbeat:
        heartbeat.start()

    def checkpoint(stage, **details):
        if not stage_hook:
            return
        directive = stage_hook(stage, phase="before", **details)
        if directive == "stop_heartbeat":
            stop_heartbeat.set()
            if heartbeat:
                heartbeat.join()
            stage_hook(stage, phase="after_heartbeat_stop", **details)

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
        checkpoint("extraction")
        metadata = extract_invoice_metadata(markdown)
        passages = chunk_text(markdown)
        passage_vectors = []
        for offset in range(0, len(passages), 32):
            passage_vectors.extend(embed_many([item["text"] for item in passages[offset:offset + 32]]))
        vector = (embed_file(data, content_type, job["id"]) if config["document_mode"] == "direct"
                  else embed(markdown))
        es = client()
        generation = uuid.uuid4().hex
        if not attempt_id or not prepare_generation(job["id"], attempt_id, generation, len(passages), db_path):
            raise RuntimeError("Worker attempt is no longer current")
        for index, (passage, passage_vector) in enumerate(zip(passages, passage_vectors)):
            if index % 32 == 0:
                _ensure_current_attempt(job, db_path)
            es.index(index=PASSAGES_INDEX, id=f"{job['id']}:{generation}:{index}", document={
                "invoice_id": job["id"], "filename": job["filename"],
                "generation": generation, "active": False, "chunk_index": index,
                "start_offset": passage["start_offset"], "end_offset": passage["end_offset"],
                "content": passage["text"], "embedding": passage_vector,
            })
            checkpoint("passage", index=index)
        _ensure_current_attempt(job, db_path)
        es.index(index=INDEX, id=f"{job['id']}:{generation}", document={
            "invoice_id": job["id"], "sha256": job["sha256"],
            "filename": job["filename"], "model": settings()["model"], "generation": generation,
            "content": markdown, **metadata, "embedding": vector,
        })
        checkpoint("document")
        _ensure_current_attempt(job, db_path)
        es.indices.refresh(index=[INDEX, PASSAGES_INDEX])
        checkpoint("refresh")
        indexed_passages = es.count(index=PASSAGES_INDEX, query={"bool": {"filter": [
            {"term": {"invoice_id": job["id"]}}, {"term": {"generation": generation}},
        ]}})["count"]
        indexed_documents = es.count(index=INDEX, query={"term": {"generation": generation}})["count"]
        if indexed_passages != len(passages) or indexed_documents != 1:
            raise RuntimeError("La generación candidata no superó la verificación de integridad")
        checkpoint("before_activation")
        if not activate_generation(job["id"], attempt_id, generation, db_path):
            raise RuntimeError("Worker attempt is no longer current")
        checkpoint("after_activation")
        status, error = "indexed", None
    except Exception as exc:
        if generation and attempt_id:
            try:
                record_publication_error(job["id"], attempt_id, db_path)
            except Exception:
                pass
        status, error = "failed", type(exc).__name__
    finally:
        stop_heartbeat.set()
        if heartbeat:
            heartbeat.join()
    with closing(connect_jobs(db_path)) as db:
        if attempt_id:
            db.execute("UPDATE jobs SET status=?,error=?,locked_at=NULL WHERE id=? AND status='processing' AND attempt_id=?",
                       (status, error, job["id"], attempt_id))
        else:
            db.execute("UPDATE jobs SET status=?,error=?,locked_at=NULL WHERE id=?", (status, error, job["id"]))
        db.commit()
    if status == "indexed" and content_type == "application/pdf":
        try:
            from application.structured_extraction import extract_document
            extract_document(job["id"], db_path=db_path)
        except Exception:
            logging.exception("Structured extraction failed for document %s", job["id"])


def _heartbeat_until_stopped(job_id, attempt_id, stop, db_path=None):
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            if not heartbeat_job(job_id, attempt_id, db_path):
                return
        except Exception:
            continue


def _ensure_current_attempt(job, db_path=None):
    if not job.get("attempt_id"):
        return
    with closing(connect_jobs(db_path)) as db:
        lease = (f"locked_at >= CURRENT_TIMESTAMP - INTERVAL '{LEASE_SECONDS} seconds'" if db.postgres else
                 f"locked_at >= datetime('now','-{LEASE_SECONDS} seconds')")
        row = db.execute(f"SELECT 1 FROM jobs WHERE id=? AND status='processing' AND attempt_id=? AND {lease}",
                         (job["id"], job["attempt_id"])).fetchone()
    if not row:
        raise RuntimeError("Worker attempt is no longer current")


def cleanup_generations(es=None, batch_size=50, *, db_path=None):
    es = es or client()
    deleted = errors = 0
    with closing(connect_jobs(db_path)) as db:
        if not lock_generation_cleanup(db):
            db.rollback()
            return {"deleted": 0, "errors": 0, "busy": True}
        for generation in stale_generations(db, batch_size):
            query = {"term": {"generation": generation["generation_id"]}}
            try:
                es.delete_by_query(index=INDEX, query=query, refresh=True)
                es.delete_by_query(index=PASSAGES_INDEX, query=query, refresh=True)
            except Exception:
                errors += 1
                continue
            if delete_generation(db, generation["generation_id"]):
                deleted += 1
            else:
                errors += 1
        record_cleanup_metrics(db, deleted, errors)
        db.commit()
    return {"deleted": deleted, "errors": errors, "busy": False}
