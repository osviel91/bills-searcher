import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import worker


job = json.loads(os.environ["PHASE0C_JOB"])
hook_dir = Path(os.environ["PHASE0C_HOOK_DIR"])
claimed_path = hook_dir / "hook-claimed.json"


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


worker.process(job, stage_hook=stage_hook)
