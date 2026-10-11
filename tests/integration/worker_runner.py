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

if os.getenv("SHADOW_TEST_MODE"):
    mode = os.environ["SHADOW_TEST_MODE"]
    original_run_isolated = shadow_extraction._run_isolated

    def run_test_child(_target, args, timeout):
        target = getattr(shadow_fixtures, f"child_{mode}")
        return original_run_isolated(target, (str(hook_dir),), timeout)

    shadow_extraction._run_isolated = run_test_child


class DiagnosticCapture:
    def emit(self, record):
        diagnostic = getattr(record, "as400_shadow", None)
        if diagnostic is None:
            return
        safe = {key: diagnostic[key] for key in (
            "status", "template", "abstention", "detector_version", "extractor_versions",
            "fields_produced", "lines_produced", "warnings", "duration_ms", "comparison",
            "error_type",
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
