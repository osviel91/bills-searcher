ALTER TABLE extraction_runs
    ADD COLUMN IF NOT EXISTS result_fingerprint TEXT;
