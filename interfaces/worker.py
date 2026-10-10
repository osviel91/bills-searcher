import os
import time
from pathlib import Path

from application import ingestion
from infrastructure.jobstore import claim_next_job, connect as connect_jobs, recover_stale_jobs
from infrastructure.object_store import ensure_container
from infrastructure.storage import client, initialize_index

DATA = Path(os.getenv("DATA_DIR", "/data"))
DB = DATA / "jobs.sqlite3"


def connect():
    return connect_jobs(DB)


def process(job, stage_hook=None):
    return ingestion.process(job, stage_hook, db_path=DB)


def cleanup_generations(es=None, batch_size=50):
    return ingestion.cleanup_generations(es, batch_size, db_path=DB)


def _ensure_current_attempt(job):
    return ingestion._ensure_current_attempt(job, DB)


def _heartbeat_until_stopped(job_id, attempt_id, stop):
    return ingestion._heartbeat_until_stopped(job_id, attempt_id, stop, DB)


def main():
    ensure_container()
    es = client()
    initialize_index(es)
    recover_stale_jobs(DB)
    cleanup_generations(es)
    next_cleanup = time.monotonic() + 60
    while True:
        if time.monotonic() >= next_cleanup:
            cleanup_generations(es)
            next_cleanup = time.monotonic() + 60
        job = claim_next_job(DB)
        if job:
            process(job)
        else:
            time.sleep(2)
