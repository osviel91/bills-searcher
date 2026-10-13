ALTER TABLE extraction_runs
    ADD COLUMN IF NOT EXISTS result_fingerprint_version INTEGER NOT NULL DEFAULT 1;
