# Phase 1 module boundaries

## Runtime map

- `domain/` contains pure invoice metadata extraction, passage splitting and evidence-ranking metrics. It has no storage or web dependencies.
- `application/search.py` coordinates hybrid invoice/evidence search and structured filtering. `application/ingestion.py` coordinates extraction, deterministic metadata, embeddings, candidate indexing, generation activation and cleanup.
- `infrastructure/` owns PostgreSQL job/lease state, Elasticsearch mappings/client, Azure Blob access and embedding-provider access.
- `interfaces/rest.py` owns the FastAPI REST surface and existing embedded web page. `interfaces/mcp.py` owns Streamable HTTP tool definitions. `interfaces/worker.py` owns the worker polling loop and process entrypoint.
- Root `app.py` and `worker.py` are compatibility entrypoints for the existing Uvicorn and Compose commands. Root infrastructure/domain module shims preserve imports used by scripts and existing consumers while production imports use the explicit packages.

## Compatibility and checks

REST paths, request/response models, MCP tool names and schemas, worker command, Compose services and the PostgreSQL/lease/generation protocol are unchanged. The deterministic test suite is `docker compose --env-file .env --env-file .env.migration run --build --rm --no-deps app python -m unittest discover -s tests -v`; the isolated real-service suite is `./scripts/run_integration.sh`.

The Software Docs shared overview still names the pre-modularization root files. It was intentionally left unchanged under the Fase 1 restriction against shared-resource edits; update it only with explicit authorization.
