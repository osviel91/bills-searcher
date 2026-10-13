ALTER TABLE extraction_runs DROP CONSTRAINT IF EXISTS extraction_runs_status_check;
ALTER TABLE extraction_runs ADD CONSTRAINT extraction_runs_status_check
    CHECK (status IN ('running','succeeded','failed','discarded_fenced'));

ALTER TABLE extraction_runs
    ADD COLUMN IF NOT EXISTS document_sha256 TEXT,
    ADD COLUMN IF NOT EXISTS attempt_id TEXT,
    ADD COLUMN IF NOT EXISTS detector_id TEXT,
    ADD COLUMN IF NOT EXISTS detector_version TEXT,
    ADD COLUMN IF NOT EXISTS template TEXT,
    ADD COLUMN IF NOT EXISTS extractor_id TEXT,
    ADD COLUMN IF NOT EXISTS extractor_set_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS execution_mode TEXT,
    ADD COLUMN IF NOT EXISTS result_schema_version INTEGER,
    ADD COLUMN IF NOT EXISTS idempotency_key TEXT,
    ADD COLUMN IF NOT EXISTS abstention_reason TEXT,
    ADD COLUMN IF NOT EXISTS diagnostics_code TEXT,
    ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS execution_slot TEXT,
    ADD COLUMN IF NOT EXISTS document_size_bytes BIGINT;

ALTER TABLE extraction_runs ADD CONSTRAINT extraction_runs_terminal_shape_check CHECK (
    execution_mode IS NULL OR
    (status='running' AND finished_at IS NULL) OR
    (status IN ('succeeded','failed','discarded_fenced') AND finished_at IS NOT NULL));
ALTER TABLE extraction_runs ADD CONSTRAINT extraction_runs_unknown_abstention_check CHECK (
    template IS DISTINCT FROM 'unknown' OR abstention_reason IS NOT NULL);

CREATE UNIQUE INDEX IF NOT EXISTS extraction_runs_document_idempotency
    ON extraction_runs(document_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS extraction_runs_document_attempt_slot
    ON extraction_runs(document_id,attempt_id,execution_slot)
    WHERE attempt_id IS NOT NULL AND execution_slot IS NOT NULL;
CREATE INDEX IF NOT EXISTS extraction_runs_logical_identity
    ON extraction_runs(document_id,document_sha256,extractor_set_fingerprint,created_at DESC);

ALTER TABLE extraction_fields
    ADD COLUMN IF NOT EXISTS observation_status TEXT,
    ADD COLUMN IF NOT EXISTS candidates_json TEXT,
    ADD CONSTRAINT extraction_fields_observation_status_check CHECK (
        observation_status IS NULL OR observation_status IN
        ('confirmed','absent','ambiguous','not_applicable','unreadable','conflict',
         'insufficient_evidence','unverified_semantics'));
ALTER TABLE extraction_evidence
    ADD COLUMN IF NOT EXISTS label_quote TEXT,
    ADD COLUMN IF NOT EXISTS label_bounds_json TEXT,
    ADD COLUMN IF NOT EXISTS method TEXT,
    ADD COLUMN IF NOT EXISTS extractor_id TEXT,
    ADD COLUMN IF NOT EXISTS extractor_version TEXT,
    ADD COLUMN IF NOT EXISTS source_token_ids_json TEXT;

CREATE OR REPLACE FUNCTION guard_final_extraction_run() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE retention_ok BOOLEAN;
BEGIN
    retention_ok := pg_catalog.pg_has_role(current_user, 'extraction_retention_operator', 'member')
        AND current_setting('app.retention_operation', true) = 'on';
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'running' AND NOT retention_ok THEN
            RAISE EXCEPTION 'extraction runs must be created in running state';
        END IF;
        RETURN NEW;
    END IF;
    IF OLD.status IN ('succeeded','failed','discarded_fenced')
       AND NOT retention_ok THEN
        RAISE EXCEPTION 'final extraction runs are immutable';
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END $$;
DROP TRIGGER IF EXISTS extraction_runs_immutable ON extraction_runs;
CREATE TRIGGER extraction_runs_immutable BEFORE INSERT OR UPDATE OR DELETE ON extraction_runs
    FOR EACH ROW EXECUTE FUNCTION guard_final_extraction_run();

CREATE OR REPLACE FUNCTION guard_final_extraction_child() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    old_status TEXT;
    new_status TEXT;
    retention_ok BOOLEAN;
BEGIN
    retention_ok := pg_catalog.pg_has_role(current_user, 'extraction_retention_operator', 'member')
        AND current_setting('app.retention_operation', true) = 'on';
    IF TG_OP <> 'INSERT' THEN
        SELECT status INTO old_status FROM extraction_runs WHERE id=OLD.run_id FOR UPDATE;
    END IF;
    IF TG_OP <> 'DELETE' THEN
        SELECT status INTO new_status FROM extraction_runs WHERE id=NEW.run_id FOR UPDATE;
    END IF;
    IF NOT retention_ok AND (old_status IN ('succeeded','failed','discarded_fenced')
        OR new_status IN ('succeeded','failed','discarded_fenced')
        OR new_status IS DISTINCT FROM 'running') THEN
        RAISE EXCEPTION 'observations may only change while their run is running';
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END $$;
DROP TRIGGER IF EXISTS extraction_pages_immutable ON extraction_pages;
CREATE TRIGGER extraction_pages_immutable BEFORE INSERT OR UPDATE OR DELETE ON extraction_pages
    FOR EACH ROW EXECUTE FUNCTION guard_final_extraction_child();
DROP TRIGGER IF EXISTS extraction_fields_immutable ON extraction_fields;
CREATE TRIGGER extraction_fields_immutable BEFORE INSERT OR UPDATE OR DELETE ON extraction_fields
    FOR EACH ROW EXECUTE FUNCTION guard_final_extraction_child();
DROP TRIGGER IF EXISTS extraction_lines_immutable ON extraction_lines;
CREATE TRIGGER extraction_lines_immutable BEFORE INSERT OR UPDATE OR DELETE ON extraction_lines
    FOR EACH ROW EXECUTE FUNCTION guard_final_extraction_child();
DROP TRIGGER IF EXISTS extraction_evidence_immutable ON extraction_evidence;
CREATE TRIGGER extraction_evidence_immutable BEFORE INSERT OR UPDATE OR DELETE ON extraction_evidence
    FOR EACH ROW EXECUTE FUNCTION guard_final_extraction_child();

CREATE OR REPLACE FUNCTION retire_extraction_run(target_id TEXT) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE run_status TEXT;
BEGIN
    PERFORM pg_catalog.set_config('app.retention_operation','on',true);
    SELECT status INTO run_status FROM public.extraction_runs WHERE id=target_id FOR UPDATE;
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    IF run_status NOT IN ('succeeded','failed','discarded_fenced') THEN
        RAISE EXCEPTION 'only a finalized extraction run can be retired';
    END IF;
    DELETE FROM public.extraction_evidence WHERE run_id=target_id;
    DELETE FROM public.extraction_fields WHERE run_id=target_id;
    DELETE FROM public.extraction_lines WHERE run_id=target_id;
    DELETE FROM public.extraction_pages WHERE run_id=target_id;
    DELETE FROM public.extraction_runs WHERE id=target_id;
    RETURN TRUE;
END $$;
ALTER FUNCTION retire_extraction_run(TEXT) OWNER TO extraction_retention_operator;
REVOKE ALL ON FUNCTION retire_extraction_run(TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION retire_extraction_run(TEXT) TO extraction_retention_operator;
