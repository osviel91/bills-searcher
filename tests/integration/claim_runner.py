import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jobstore import claim_next_job


start = Path(os.environ["PHASE0C_HOOK_DIR"]) / "claim-start"
while not start.exists():
    time.sleep(0.005)
job = claim_next_job()
print(f"{job['id']}|{job['attempt_id']}" if job else "")
