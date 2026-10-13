import os
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from infrastructure import jobstore
from infrastructure.migrations import apply
import migrate_schema


@unittest.skipUnless(os.getenv("RUN_POSTGRES_CONTRACT_TESTS") == "1", "requires real PostgreSQL")
class PostgresPersistenceContractTests(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self.blocker_table = f"retention_blocker_{uuid.uuid4().hex}"
        self.field_id = "total"
        self.page = 1
        with jobstore.connect() as db:
            db.execute("INSERT INTO jobs(id,sha256,status,attempt_id) VALUES(?,?,?,?)",
                       (self.job_id, f"sha-{self.job_id}", "indexed", "attempt-old"))
            db.execute("""INSERT INTO extraction_runs
                (id,document_id,extractor_version,status,warnings_json,attempt_id,idempotency_key,execution_mode)
                VALUES(?,?,?,'running','[]','attempt-old',?,'specialized')""",
                (self.run_id, self.job_id, "contract-v1", f"key-{self.run_id}"))
            db.execute("INSERT INTO extraction_pages(run_id,page_number,structure_json) VALUES(?,?,?)",
                       (self.run_id, self.page, "{}"))
            db.execute("INSERT INTO extraction_fields(run_id,name,value_json,warnings_json) VALUES(?,?,?,?)",
                       (self.run_id, self.field_id, '"42.00"', "[]"))
            db.execute("""INSERT INTO extraction_lines
                (run_id,line_number,description,values_json,warnings_json) VALUES(?,1,'line','{}','[]')""",
                (self.run_id,))
            db.execute("""INSERT INTO extraction_evidence
                (id,run_id,field_name,page_number,quote) VALUES(?,?,?,?,'synthetic')""",
                (str(uuid.uuid4()), self.run_id, self.field_id, self.page))
            db.execute("UPDATE extraction_runs SET status='succeeded',finished_at=CURRENT_TIMESTAMP WHERE id=?",
                       (self.run_id,))

    def tearDown(self):
        with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL ROLE extraction_retention_operator")
                conn.execute("SELECT retire_extraction_run(%s)", (self.run_id,))
                conn.execute("RESET ROLE")
                conn.execute("DELETE FROM jobs WHERE id=%s", (self.job_id,))

    def _app_rejects(self, statement, params=()):
        with psycopg.connect(jobstore.DATABASE_URL) as conn:
            with self.assertRaises(psycopg.Error):
                conn.execute(statement, params)

    def test_runtime_role_is_not_owner_admin_or_retention_member(self):
        with psycopg.connect(jobstore.DATABASE_URL, row_factory=dict_row) as conn:
            privileges = conn.execute("""SELECT r.rolsuper, r.rolinherit,
                    pg_catalog.pg_has_role(current_user,'extraction_retention_operator','member') AS can_retain,
                    pg_catalog.has_schema_privilege(current_user,'public','CREATE') AS can_create,
                    pg_catalog.pg_get_userbyid(c.relowner)=current_user AS owns_runs
                FROM pg_roles r JOIN pg_class c ON c.relname='extraction_runs'
                JOIN pg_namespace n ON n.oid=c.relnamespace AND n.nspname='public'
                WHERE r.rolname=current_user""").fetchone()
        self.assertFalse(privileges["rolsuper"])
        self.assertFalse(privileges["rolinherit"])
        self.assertFalse(privileges["can_retain"])
        self.assertFalse(privileges["can_create"])
        self.assertFalse(privileges["owns_runs"])

    def test_custom_setting_does_not_bypass_terminal_immutability(self):
        with psycopg.connect(jobstore.DATABASE_URL) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL app.retention_operation='on'")
                with self.assertRaises(psycopg.Error):
                    conn.execute("UPDATE extraction_runs SET error='overwrite' WHERE id=%s", (self.run_id,))
            with conn.transaction():
                conn.execute("SET LOCAL app.retention_operation='on'")
                with self.assertRaises(psycopg.Error):
                    conn.execute("INSERT INTO extraction_fields "
                                 "(run_id,name,value_json,warnings_json) VALUES(%s,'extra','null','[]')",
                                 (self.run_id,))

    def test_runtime_cannot_assume_retention_role(self):
        self._app_rejects("SET ROLE extraction_retention_operator")
        self._app_rejects("SELECT retire_extraction_run(%s)", (self.run_id,))

    def test_runtime_cannot_insert_update_or_delete_any_finalized_aggregate_row(self):
        self._app_rejects("INSERT INTO extraction_runs "
                          "(id,document_id,extractor_version,status,warnings_json) "
                          "VALUES(%s,%s,'v1','succeeded','[]')", (str(uuid.uuid4()), self.job_id))
        self._app_rejects("UPDATE extraction_runs SET error='overwrite' WHERE id=%s", (self.run_id,))
        self._app_rejects("DELETE FROM extraction_runs WHERE id=%s", (self.run_id,))

        self._app_rejects("INSERT INTO extraction_pages(run_id,page_number,structure_json) VALUES(%s,2,'{}')",
                          (self.run_id,))
        self._app_rejects("UPDATE extraction_pages SET structure_json='null' WHERE run_id=%s", (self.run_id,))
        self._app_rejects("DELETE FROM extraction_pages WHERE run_id=%s", (self.run_id,))

        self._app_rejects("INSERT INTO extraction_fields(run_id,name,value_json,warnings_json) "
                          "VALUES(%s,'extra','null','[]')", (self.run_id,))
        self._app_rejects("UPDATE extraction_fields SET value_json='null' WHERE run_id=%s", (self.run_id,))
        self._app_rejects("DELETE FROM extraction_fields WHERE run_id=%s", (self.run_id,))

        self._app_rejects("INSERT INTO extraction_lines(run_id,line_number,values_json,warnings_json) "
                          "VALUES(%s,2,'{}','[]')", (self.run_id,))
        self._app_rejects("UPDATE extraction_lines SET values_json='null' WHERE run_id=%s", (self.run_id,))
        self._app_rejects("DELETE FROM extraction_lines WHERE run_id=%s", (self.run_id,))

        self._app_rejects("INSERT INTO extraction_evidence "
                          "(id,run_id,field_name,page_number,quote) VALUES(%s,%s,%s,1,'extra')",
                          (str(uuid.uuid4()), self.run_id, self.field_id))
        self._app_rejects("UPDATE extraction_evidence SET quote='overwrite' WHERE run_id=%s", (self.run_id,))
        self._app_rejects("DELETE FROM extraction_evidence WHERE run_id=%s", (self.run_id,))

    def test_retention_function_requires_administrative_role_and_removes_the_whole_run(self):
        with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL ROLE extraction_retention_operator")
                self.assertTrue(conn.execute("SELECT retire_extraction_run(%s)", (self.run_id,)).fetchone()[0])
        with psycopg.connect(jobstore.DATABASE_URL) as conn:
            for table in ("extraction_runs", "extraction_pages", "extraction_fields",
                          "extraction_lines", "extraction_evidence"):
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table} WHERE "
                                               f"{'id' if table == 'extraction_runs' else 'run_id'}=%s",
                                               (self.run_id,)).fetchone()[0], 0)

    def test_failed_administrative_retention_rolls_back_all_child_deletes(self):
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            with psycopg.connect(os.environ["DATABASE_ADMIN_URL"]) as conn:
                with conn.transaction():
                    conn.execute(sql.SQL("CREATE TABLE public.{} "
                                         "(run_id TEXT REFERENCES public.extraction_runs(id))")
                                 .format(sql.Identifier(self.blocker_table)))
                    conn.execute(sql.SQL("INSERT INTO public.{} VALUES(%s)")
                                 .format(sql.Identifier(self.blocker_table)), (self.run_id,))
                    conn.execute("SET LOCAL ROLE extraction_retention_operator")
                    conn.execute("SELECT retire_extraction_run(%s)", (self.run_id,))
        with psycopg.connect(jobstore.DATABASE_URL) as conn:
            for table in ("extraction_runs", "extraction_pages", "extraction_fields",
                          "extraction_lines", "extraction_evidence"):
                key = "id" if table == "extraction_runs" else "run_id"
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table} WHERE {key}=%s",
                                               (self.run_id,)).fetchone()[0], 1)

    def test_migration_runner_is_repeatable_under_concurrent_admin_connections(self):
        errors = []

        def migrate():
            try:
                with psycopg.connect(os.environ["DATABASE_ADMIN_URL"], row_factory=dict_row) as raw:
                    apply(jobstore.Connection(raw, True))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=migrate) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])

    def test_migrator_aborts_before_role_changes_when_runtime_owns_a_table(self):
        admin_url = os.environ["DATABASE_ADMIN_URL"]
        app_role = os.environ["APP_DB_USER"]
        app_password = os.environ["APP_DB_PASSWORD"]
        with psycopg.connect(admin_url) as conn:
            before = conn.execute("SELECT rolsuper,rolinherit,rolcanlogin FROM pg_roles WHERE rolname=%s",
                                  (app_role,)).fetchone()
            conn.execute("GRANT CREATE ON SCHEMA public TO integration_app")
            conn.execute("ALTER TABLE public.generations OWNER TO " + app_role)
        try:
            with patch.dict(os.environ, {"DATABASE_MIGRATION_URL": admin_url,
                                         "APP_DB_USER": app_role,
                                         "APP_DB_PASSWORD": app_password}):
                with self.assertRaisesRegex(RuntimeError, "owns database/schema objects"):
                    migrate_schema.main()
            with psycopg.connect(admin_url) as conn:
                after = conn.execute("SELECT rolsuper,rolinherit,rolcanlogin FROM pg_roles WHERE rolname=%s",
                                     (app_role,)).fetchone()
                owner = conn.execute("SELECT pg_get_userbyid(relowner) FROM pg_class "
                                     "WHERE relname='generations' AND relnamespace='public'::regnamespace").fetchone()[0]
            self.assertEqual(after, before)
            self.assertEqual(owner, app_role)
        finally:
            with psycopg.connect(admin_url) as conn:
                conn.execute("ALTER TABLE public.generations OWNER TO integration_admin")
                conn.execute("REVOKE CREATE ON SCHEMA public FROM integration_app")


if __name__ == "__main__":
    unittest.main()
