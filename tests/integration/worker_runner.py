import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import worker
from application import shadow_extraction
import shadow_fixtures


job = json.loads(os.environ["PHASE0C_JOB"])
hook_dir = Path(os.environ["PHASE0C_HOOK_DIR"])
claimed_path = hook_dir / "hook-claimed.json"

expected_shadow = os.getenv("SHADOW_EXPECT_ENABLED", os.getenv("AS400_SHADOW_ENABLED", "false")) == "true"
if shadow_extraction.enabled() != expected_shadow:
    raise RuntimeError("shadow runner configuration mismatch")
expected_persistence = os.getenv("SHADOW_EXPECT_PERSISTENCE", "false") == "true"
if shadow_extraction.persistence_enabled() != expected_persistence:
    raise RuntimeError("shadow persistence runner configuration mismatch")
if os.getenv("SHADOW_TEST_TIMEOUT_SECONDS"):
    shadow_extraction.TIMEOUT_SECONDS = float(os.environ["SHADOW_TEST_TIMEOUT_SECONDS"])


def run_inline(target, args, _timeout):
    class InlineSender:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

        def close(self):
            pass

    sender = InlineSender()
    target(sender, *args)
    return sender.messages[-1]


if os.getenv("SHADOW_TEST_MODE"):
    mode = os.environ["SHADOW_TEST_MODE"]
    if mode in {"failure", "timeout", "stale"}:
        original_run_isolated = shadow_extraction._run_isolated

        def run_test_child(_target, args, timeout):
            target = getattr(shadow_fixtures, f"child_{mode}")
            return original_run_isolated(target, (str(hook_dir),), timeout)

        shadow_extraction._run_isolated = run_test_child
    elif mode == "adapter-failure":
        from infrastructure import specialized_extraction_writer
        from application.template_detection import TemplateDetection
        from domain.structured_extraction import ExtractedField
        from infrastructure.specialized_extraction_writer import Component, serialize_shadow_extraction_result

        def fail_adapter(*_args, **_kwargs):
            raise ValueError("synthetic adapter failure")

        specialized_extraction_writer.adapt_template_result = fail_adapter

        extraction = {"classification": TemplateDetection("T01", "as400-t01", "1", {}, None),
                      "extractor_id": "as400-t01-header-summary", "extractor_version": "1",
                      "line_extractor": Component("as400-t01-lines", "1"),
                      "fields": (ExtractedField("invoice_number", "SYNTHETIC", status="confirmed"),),
                      "lines": (), "warnings": (), "status": "succeeded"}

        def return_prepared_result(_target, _args, _timeout):
            return {"status": "succeeded", "template": "T01", "fields_produced": 1,
                    "lines_produced": 0, "warnings": 0, "extraction_duration_ms": 1,
                    "persistence_status": "prepared",
                    "persistence_payload": serialize_shadow_extraction_result(extraction)}

        shadow_extraction._run_isolated = return_prepared_result
    elif mode == "extractor-error":
        from application.template_detection import TemplateDetection
        from infrastructure.specialized_extraction_writer import Component, serialize_shadow_extraction_result

        extraction = {"classification": TemplateDetection("T01", "as400-t01", "1", {}, None),
                      "extractor_id": "as400-t01-header-summary", "extractor_version": "1",
                      "line_extractor": Component("as400-t01-lines", "1"),
                      "fields": (), "lines": (), "warnings": (), "status": "failed",
                      "error": "ValueError", "diagnostics_code": "shadow-extractor-error"}

        def return_failed_result(_target, _args, _timeout):
            return {"status": "extractor_error", "template": "T01", "fields_produced": 0,
                    "lines_produced": 0, "warnings": 0, "extraction_duration_ms": 1,
                    "persistence_status": "prepared",
                    "persistence_payload": serialize_shadow_extraction_result(extraction)}

        shadow_extraction._run_isolated = return_failed_result
    elif mode == "commit-ack-lost":
        import psycopg
        from infrastructure import specialized_extraction_writer

        original_connect = specialized_extraction_writer.connect

        class LoseCommitAck:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, exc_type, exc, traceback):
                result = self.connection.__exit__(exc_type, exc, traceback)
                if exc_type is None:
                    raise psycopg.OperationalError("synthetic lost acknowledgement after COMMIT")
                return result

            def __getattr__(self, name):
                return getattr(self.connection, name)

        specialized_extraction_writer.connect = lambda path=None: LoseCommitAck(original_connect(path))
        shadow_extraction._run_isolated = run_inline
class DiagnosticCapture:
    def emit(self, record):
        diagnostic = getattr(record, "as400_shadow", None)
        if diagnostic is None:
            return
        safe = {key: diagnostic[key] for key in (
            "status", "template", "abstention", "detector_version", "extractor_versions",
            "fields_produced", "lines_produced", "warnings", "duration_ms",
            "extraction_duration_ms", "shadow_duration_ms", "comparison",
            "error_type", "document_id", "attempt_id", "shadow_status",
            "persistence_enabled", "persistence_status",
            "persistence_error_type", "persistence_error_code", "persistence_duration_ms",
            "persistence_run_id", "persistence_attempt_id", "idempotency_key",
        ) if key in diagnostic}
        (hook_dir / f"diagnostic-{job['id']}.json").write_text(json.dumps(safe))


capture = logging.Handler()
capture.emit = DiagnosticCapture().emit
shadow_extraction.LOGGER.addHandler(capture)
shadow_extraction.LOGGER.setLevel(logging.INFO)


def stage_hook(stage, phase="before", **details):
    try:
        config = json.loads((hook_dir / "hook.json").read_text())
    except (OSError, json.JSONDecodeError):
        return
    if (config.get("job_id") not in (None, job["id"]) or
            config.get("attempt_id") not in (None, job.get("attempt_id")) or
            config.get("stage") != stage or
            config.get("index") is not None and config["index"] != details.get("index")):
        return
    if phase == "before":
        claimed_path.write_text(json.dumps({"job_id": job["id"], "attempt_id": job.get("attempt_id")}))
        if config.get("stop_heartbeat"):
            return "stop_heartbeat"
    if phase == "after_heartbeat_stop" or not config.get("stop_heartbeat"):
        (hook_dir / "hook-reached").write_text(json.dumps({"stage": stage, **details}))
        if config.get("mode") == "fail":
            raise RuntimeError("integration failpoint")
        while not (hook_dir / "hook-release").exists():
            time.sleep(0.02)


if __name__ == "__main__":
    worker.process(job, stage_hook=stage_hook)
    (hook_dir / f"runner-complete-{job['id']}.json").write_text(
        json.dumps({"active_children": len(multiprocessing.active_children())}))
