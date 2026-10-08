# Agent Instructions

- This is a single Python 3.12 app, not a package workspace. `app.py` serves the FastAPI/MCP API; `worker.py` claims queued jobs, extracts/embeds files, and indexes them. Supporting boundaries are `jobstore.py` (PostgreSQL, with SQLite fallback for local tests), `object_store.py`, `embeddings.py`, and `storage.py` (Elasticsearch).
- Run deterministic tests with `docker compose run --rm --no-deps app python -m unittest discover -s tests -v`. Focus one test file with `-p 'test_jobstore.py'` (replace with the relevant filename). `scripts/smoke_search.py` and `scripts/evaluate_search.py` call a live service; evaluation requires a human-labeled JSON case file. No lint or typecheck command is configured.
- Compose loads `.env` and `.env.runtime`; copy their `.example` files for local setup and keep real credentials out of source control. Do not send real invoices to an embedding endpoint until its location and data controls are verified; use synthetic/anonymous documents for remote endpoints.
- Embedding model, provider, document mode, and vector dimensions must agree with the Elasticsearch index mapping. Changing model/dimensions requires a new index and reindexing; the app rejects a dimension mismatch.
- `docker compose down -v` deletes local PostgreSQL, Azurite, and Elasticsearch data. The README is the source for setup, API, and deployment details; `docker-compose.portainer.yml` is for Portainer deployment, not local testing.

## Software Docs architecture

- The Software Docs project for this repository is `Búsqueda semántica local de facturas` (`01a11d13-98db-75a9-a819-84e34ef5a758`); use this project ID for Software Docs operations.
- For every proposed feature, behavior change, or refactor, inspect the Software Docs project and assess which existing architectural artifacts must change and which new artifacts are needed to keep the code and the project's holistic view aligned. Choose representations by evidence and purpose; do not add diagrams mechanically or invent unknown behavior.
- Make documentation changes in an owned MY WORK context, validate them, then submit the selected resources as an Architectural Proposal. Never edit SHARED directly or treat submission, review, or preview as publication; promotion is a separate explicit step.
