import os

import psycopg
from psycopg import sql


def main():
    admin_url = os.environ["DATABASE_MIGRATION_URL"]
    app_role = os.environ["APP_DB_USER"]
    app_password = os.environ["APP_DB_PASSWORD"]

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin_role = admin.execute("SELECT current_user").fetchone()[0]
        if admin_role == app_role:
            raise RuntimeError("migration and runtime database roles must be distinct")
        owns_objects = admin.execute("""SELECT
            EXISTS (SELECT 1 FROM pg_database WHERE datname=current_database()
                    AND pg_catalog.pg_get_userbyid(datdba)=%s)
            OR EXISTS (SELECT 1 FROM pg_namespace WHERE nspname='public'
                       AND pg_catalog.pg_get_userbyid(nspowner)=%s)
            OR EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S','f')
                       AND pg_catalog.pg_get_userbyid(c.relowner)=%s)
            OR EXISTS (SELECT 1 FROM pg_proc WHERE proname IN
                       ('guard_final_extraction_run','guard_final_extraction_child','retire_extraction_run')
                       AND pg_catalog.pg_get_userbyid(proowner)=%s)""",
            (app_role, app_role, app_role, app_role)).fetchone()[0]
        if owns_objects:
            raise RuntimeError("runtime database role owns database/schema objects; provision a separate runtime role")
        if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname='extraction_retention_operator'").fetchone():
            admin.execute("CREATE ROLE extraction_retention_operator NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION")
        admin.execute("ALTER ROLE extraction_retention_operator NOLOGIN NOINHERIT NOSUPERUSER "
                      "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS")
        operator_memberships = admin.execute("""SELECT granted.rolname FROM pg_auth_members m
            JOIN pg_roles granted ON granted.oid=m.roleid
            JOIN pg_roles member ON member.oid=m.member
            WHERE member.rolname='extraction_retention_operator'""").fetchall()
        for membership in operator_memberships:
            admin.execute(sql.SQL("REVOKE {} FROM extraction_retention_operator")
                          .format(sql.Identifier(membership[0])))
        if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (app_role,)).fetchone():
            admin.execute(sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(sql.Identifier(app_role), sql.Literal(app_password)))
        admin.execute(sql.SQL(
            "ALTER ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}"
        ).format(sql.Identifier(app_role), sql.Literal(app_password)))
        memberships = admin.execute("""SELECT granted.rolname FROM pg_auth_members m
            JOIN pg_roles granted ON granted.oid=m.roleid
            JOIN pg_roles member ON member.oid=m.member
            WHERE member.rolname=%s""", (app_role,)).fetchall()
        for membership in memberships:
            admin.execute(sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(membership[0]), sql.Identifier(app_role)))
        admin.execute("GRANT USAGE, CREATE ON SCHEMA public TO extraction_retention_operator")
        admin.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        admin.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(sql.Identifier(app_role)))

    os.environ["SCHEMA_INITIALIZER"] = "true"
    from infrastructure import jobstore

    jobstore.DATABASE_URL = admin_url
    jobstore._initialized.discard(admin_url)
    with jobstore.connect() as db:
        db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        db.execute("REVOKE CREATE ON SCHEMA public FROM extraction_retention_operator")
        database_name = db.execute("SELECT current_database()").fetchone()["current_database"]
        db.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(database_name), sql.Identifier(app_role)))
        db.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(app_role)))
        for table, privileges in (
            ("jobs", "SELECT, INSERT, UPDATE"),
            ("generations", "SELECT, INSERT, UPDATE, DELETE"),
            ("maintenance_metrics", "SELECT, INSERT, UPDATE"),
            ("extraction_runs", "SELECT, INSERT, UPDATE"),
            ("extraction_pages", "SELECT, INSERT, UPDATE"),
            ("extraction_fields", "SELECT, INSERT, UPDATE"),
            ("extraction_lines", "SELECT, INSERT, UPDATE"),
            ("extraction_evidence", "SELECT, INSERT, UPDATE"),
        ):
            db.execute(sql.SQL("GRANT " + privileges + " ON TABLE public.{} TO {}").format(
                sql.Identifier(table), sql.Identifier(app_role)))
        for table in ("extraction_runs", "extraction_pages", "extraction_fields",
                      "extraction_lines", "extraction_evidence"):
            privileges = "SELECT, UPDATE, DELETE" if table == "extraction_runs" else "SELECT, DELETE"
            db.execute(sql.SQL("GRANT " + privileges + " ON TABLE public.{} TO extraction_retention_operator")
                       .format(sql.Identifier(table)))
        db.execute("GRANT USAGE ON SCHEMA public TO extraction_retention_operator")


if __name__ == "__main__":
    main()
