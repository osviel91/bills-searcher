import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import worker
from jobstore import claim_next_job


HOOK_DIR = Path(os.environ.get("PHASE0C_HOOK_DIR", "/integration-hooks"))


def stage_hook(job):
    claimed_path = HOOK_DIR / "hook-claimed.json"

    def hook(stage, phase="before", **details):
        config_path = HOOK_DIR / "hook.json"
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if (config.get("job_id") not in (None, job["id"]) or
                config.get("attempt_id") not in (None, job.get("attempt_id")) or
                config.get("stage") != stage or
                config.get("index") is not None and config["index"] != details.get("index")):
            return
        if phase == "before":
            try:
                with claimed_path.open("x") as claimed:
                    json.dump({"job_id": job["id"], "attempt_id": job.get("attempt_id"),
                               "stage": stage}, claimed)
            except FileExistsError:
                try:
                    claimed = json.loads(claimed_path.read_text())
                except (OSError, json.JSONDecodeError):
                    return
                if claimed.get("attempt_id") != job.get("attempt_id"):
                    return
            if config.get("stop_heartbeat"):
                return "stop_heartbeat"
        if phase == "after_heartbeat_stop" or not config.get("stop_heartbeat"):
            (HOOK_DIR / "hook-reached").write_text(json.dumps({"stage": stage, **details}))
            if config.get("mode") == "fail":
                raise RuntimeError("integration failpoint")
            while not (HOOK_DIR / "hook-release").exists():
                time.sleep(0.02)
    return hook


while not (HOOK_DIR / "workers-start").exists():
    time.sleep(0.05)

(HOOK_DIR / f"worker-ready-{socket.gethostname()}").touch()

while True:
    if (HOOK_DIR / "pause-claiming").exists():
        time.sleep(0.05)
        continue
    job = claim_next_job()
    if job:
        worker.process(job, stage_hook=stage_hook(job))
    else:
        time.sleep(0.1)
