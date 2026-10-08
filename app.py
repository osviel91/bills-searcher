import hashlib
import base64
import binascii
import hmac
import os
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from mcp.server.fastmcp import FastMCP
from starlette.responses import StreamingResponse

from embeddings import MODEL, available_models, embed, health as embedding_health, settings as embedding_settings, update_settings
from jobstore import connect as connect_jobs
from object_store import container_client, delete as delete_object, ensure_container, get as get_object, put as put_object
from storage import INDEX, client as es, initialize_index

DATA = Path(os.getenv("DATA_DIR", "/data"))
DB = DATA / "jobs.sqlite3"
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "100"))
API_TOKEN = os.getenv("API_TOKEN", "")
bearer_scheme = HTTPBearer(auto_error=False)


def validate_token(token: str | None):
    if not API_TOKEN:
        raise HTTPException(503, "API_TOKEN no está configurado")
    if not token or not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(401, "Token no válido", headers={"WWW-Authenticate": "Bearer"})


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    validate_token(credentials.credentials if credentials else None)


class McpTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/mcp"):
            try:
                authorization = request.headers.get("authorization", "")
                scheme, _, token = authorization.partition(" ")
                validate_token(token if scheme.lower() == "bearer" else None)
            except HTTPException as error:
                return JSONResponse({"detail": error.detail}, status_code=error.status_code,
                                    headers=error.headers)
        return await call_next(request)


mcp = FastMCP("Invoice indexing and semantic search", stateless_http=True, json_response=True)


@asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="Invoice semantic search API",
    summary="Ingesta asíncrona y búsqueda híbrida de facturas PDF.",
    description=(
        "La ingestión recibe bloques de PDFs y devuelve un resultado por fichero. "
        "Los documentos se extraen e indexan en segundo plano; consulta su estado por ID. "
        "La búsqueda combina texto y vectores, y su puntuación es un ranking relativo. "
        "La API requiere `Authorization: Bearer <token>` salvo `/api/health`. "
        "El servidor MCP Streamable HTTP está disponible en `/mcp`; sus herramientas "
        "son `index_invoice`, `get_invoice_status` y `semantic_search`."
    ),
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {"name": "Facturas", "description": "Carga por bloques, consulta de estado, reintento y descarga de PDFs."},
        {"name": "Búsqueda", "description": "Búsqueda híbrida léxica y semántica sobre facturas indexadas."},
        {"name": "Sistema", "description": "Estado de las dependencias del servicio."},
    ],
    lifespan=lifespan,
)
app.add_middleware(McpTokenMiddleware)


def normalize_pdf(data: bytes) -> bytes:
    # PDF files may have a short wrapper/preamble; the header is allowed within the first 1024 bytes.
    signature = data.find(b"%PDF-", 0, 1024)
    if signature < 0:
        raise ValueError("El fichero no contiene una cabecera PDF válida en los primeros 1024 bytes")
    return data[signature:]


def connect():
    return connect_jobs(DB)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Concepto o descripción que se quiere encontrar en las facturas.", examples=["mantenimiento de impresoras"])
    limit: int = Field(default=10, ge=1, le=50, description="Máximo de resultados que se devolverán.")


class EmbeddingSettings(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=2000)


class ApiError(BaseModel):
    detail: str


class UploadDocumentResult(BaseModel):
    id: str | None = Field(default=None, description="ID para consultar el estado del documento.")
    filename: str | None = Field(default=None, description="Nombre original del archivo recibido.")
    status: Literal["queued", "duplicate", "rejected", "failed"] = Field(description="Resultado individual del procesamiento de la petición.")
    error: str | None = Field(default=None, description="Motivo de rechazo o fallo, si aplica.")


class UploadResponse(BaseModel):
    documents: list[UploadDocumentResult] = Field(description="Un resultado por cada archivo enviado.")
    accepted: int = Field(description="Archivos nuevos añadidos a la cola.")
    duplicates: int = Field(description="Archivos ya registrados por su hash.")
    rejected: int = Field(description="Archivos rechazados durante la validación.")


class InvoiceStatusResponse(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "indexed", "failed"]
    error: str | None = None
    created_at: datetime


class DocumentPage(BaseModel):
    documents: list[InvoiceStatusResponse]
    total: int
    limit: int
    offset: int


class SearchResult(BaseModel):
    invoice_id: str
    filename: str
    score: float = Field(description="Puntuación RRF relativa, no una probabilidad.")
    file_url: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class HealthResponse(BaseModel):
    elasticsearch: bool
    database: bool
    object_storage: bool
    embedding_service: bool
    configured_model: str
    model_available: bool
    document_processing: str
    pdf_representation: str


def reciprocal_rank_fusion(*rankings, limit):
    fused = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            item = fused.setdefault(hit["_id"], {"_source": hit["_source"], "_score": 0.0})
            item["_score"] += 1 / (60 + rank)
    return sorted(fused.values(), key=lambda item: item["_score"], reverse=True)[:limit]


@app.on_event("startup")
def startup():
    with closing(connect()):
        pass
    ensure_container()
    try:
        initialize_index(es())
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/health", tags=["Sistema"], summary="Comprobar el estado de las dependencias",
         response_model=HealthResponse)
def health():
    try:
        es_ok = es().ping()
    except Exception:
        es_ok = False
    try:
        with closing(connect()) as db:
            db.execute("SELECT 1")
        database_ok = True
    except Exception:
        database_ok = False
    try:
        container_client().get_container_properties()
        object_store_ok = True
    except Exception:
        object_store_ok = False
    inference_ok, model_available = embedding_health()
    return {"elasticsearch": es_ok, "database": database_ok, "object_storage": object_store_ok,
            "embedding_service": inference_ok,
            "configured_model": embedding_settings()["model"],
            "model_available": model_available,
            "document_processing": "MarkItDown text extraction; OCR disabled; scanned PDFs may produce no text",
            "pdf_representation": "text-derived embeddings, not multimodal PDF embeddings"}


@app.get("/api/settings/embeddings", dependencies=[Depends(require_token)])
def get_embedding_settings():
    config = embedding_settings()
    return {"base_url": config["base_url"], "model": config["model"], "api_key_configured": bool(config["api_key"])}


@app.put("/api/settings/embeddings", dependencies=[Depends(require_token)])
def put_embedding_settings(config: EmbeddingSettings):
    current = embedding_settings()
    update_settings(config.base_url, config.model, config.api_key or current["api_key"])
    return get_embedding_settings()


@app.get("/api/settings/embeddings/models", dependencies=[Depends(require_token)])
def get_embedding_models():
    try:
        return {"models": available_models()}
    except Exception as error:
        raise HTTPException(502, f"No se pudieron detectar los modelos: {type(error).__name__}") from error


@app.post(
    "/api/documents",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Subir un bloque de facturas PDF",
    description=("Envía uno o más archivos en el campo multipart repetido `files`. "
                 "Cada archivo recibe su propio estado; un PDF inválido no cancela los demás. "
                 "Límites configurables: `MAX_BATCH_FILES` por bloque y `MAX_UPLOAD_MB` por PDF."),
    response_model=UploadResponse,
    responses={
        401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
        413: {"model": ApiError, "description": "El bloque supera el número máximo de archivos."},
        503: {"model": ApiError, "description": "El token no está configurado."},
    },
)
async def upload(files: list[UploadFile] = File(..., description="Uno o más PDFs; repite el campo `files` para cada archivo.")):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"El bloque supera el límite de {MAX_BATCH_FILES} ficheros")
    created = []
    with closing(connect()) as db:
        for file in files:
            try:
                if not file.filename or not file.filename.lower().endswith(".pdf"):
                    raise ValueError("Solo se admiten ficheros PDF")
                data = await file.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    created.append({"filename": file.filename, "status": "rejected", "error": "El PDF supera el límite configurado"})
                    continue
                data = normalize_pdf(data)
            except ValueError as error:
                created.append({"filename": file.filename, "status": "rejected", "error": str(error)})
                continue
            digest = hashlib.sha256(data).hexdigest()
            row = db.execute("SELECT id FROM jobs WHERE sha256=?", (digest,)).fetchone()
            if row:
                created.append({"id": row["id"], "filename": file.filename, "status": "duplicate"})
                continue
            doc_id = str(uuid.uuid4())
            object_key = f"documents/{doc_id}.pdf"
            stored = False
            try:
                await run_in_threadpool(put_object, object_key, data)
                stored = True
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", (doc_id, digest, file.filename, object_key, "queued"))
                db.commit()
            except Exception as error:
                db.rollback()
                if stored:
                    try:
                        delete_object(object_key)
                    except Exception:
                        pass
                if "unique" in str(error).lower():
                    row = db.execute("SELECT id FROM jobs WHERE sha256=?", (digest,)).fetchone()
                    created.append({"id": row["id"], "filename": file.filename, "status": "duplicate"})
                    continue
                created.append({"filename": file.filename, "status": "failed", "error": str(error)[:300]})
                continue
            created.append({"id": doc_id, "filename": file.filename, "status": "queued"})
    return {"documents": created,
            "accepted": sum(item["status"] == "queued" for item in created),
            "duplicates": sum(item["status"] == "duplicate" for item in created),
            "rejected": sum(item["status"] == "rejected" for item in created)}


@app.get(
    "/api/documents",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Listar facturas de forma paginada",
    response_model=DocumentPage,
    responses={401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
               503: {"model": ApiError, "description": "El token no está configurado."}},
)
def documents(limit: int = Query(default=50, ge=1, le=200, description="Cantidad máxima de documentos devueltos."),
              offset: int = Query(default=0, ge=0, description="Número de documentos que se omiten."),
              status: str | None = Query(default=None, pattern="^(queued|processing|indexed|failed)$",
                                          description="Filtra por estado de indexación.")):
    with closing(connect()) as db:
        where, params = (" WHERE status=?", [status]) if status else ("", [])
        total = db.execute("SELECT COUNT(*) AS total FROM jobs" + where, params).fetchone()["total"]
        rows = db.execute("SELECT id,filename,status,error,created_at FROM jobs" + where +
                          " ORDER BY created_at DESC,id LIMIT ? OFFSET ?", (*params, limit, offset))
        return {"documents": [dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}


@app.get(
    "/api/documents/{doc_id}",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Consultar el estado de una factura",
    response_model=InvoiceStatusResponse,
    responses={401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
               404: {"model": ApiError, "description": "No existe una factura con ese ID."},
               503: {"model": ApiError, "description": "El token no está configurado."}},
)
def document_status(doc_id: str):
    with closing(connect()) as db:
        row = db.execute("SELECT id,filename,status,error,created_at FROM jobs WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Documento no encontrado")
    return dict(row)


@app.post(
    "/api/documents/{doc_id}/retry",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Reencolar una factura fallida",
    responses={401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
               404: {"model": ApiError, "description": "No existe una factura fallida con ese ID."},
               503: {"model": ApiError, "description": "El token no está configurado."}},
)
def retry(doc_id: str):
    with closing(connect()) as db:
        result = db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=? AND status='failed'", (doc_id,))
        db.commit()
        if not result.rowcount:
            raise HTTPException(404, "Documento fallido no encontrado")
    return {"id": doc_id, "status": "queued"}


@app.get(
    "/api/documents/{doc_id}/file",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Descargar el PDF original",
    responses={
        200: {"description": "PDF original.", "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}},
        401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
        404: {"model": ApiError, "description": "No se encontró el documento o su original."},
        503: {"model": ApiError, "description": "El token no está configurado."},
    },
)
def document_file(doc_id: str):
    with closing(connect()) as db:
        row = db.execute("SELECT object_key,filename FROM jobs WHERE id=?", (doc_id,)).fetchone()
    if not row or not row["object_key"]:
        raise HTTPException(404, "PDF no encontrado")
    try:
        body = get_object(row["object_key"])
    except Exception as error:
        raise HTTPException(404, "PDF no encontrado en almacenamiento") from error

    def chunks():
        yield from body.chunks()

    return StreamingResponse(chunks(), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(row['filename'])}"})


@app.post(
    "/api/search",
    dependencies=[Depends(require_token)],
    tags=["Búsqueda"],
    summary="Buscar facturas por texto y significado",
    description="La búsqueda híbrida fusiona resultados léxicos y vectoriales; `score` sirve para ordenar y no es una probabilidad.",
    response_model=SearchResponse,
    responses={401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
               503: {"model": ApiError, "description": "El proveedor de embeddings o Elasticsearch no respondió."}},
)
def search(request: SearchRequest):
    return run_search(request.query, request.limit)


def run_search(query: str, limit: int):
    try:
        vector = embed(query, query=True)
        es_client = es()
        window = max(50, limit * 10)
        lexical = es_client.search(index=INDEX, size=window, query={"bool": {"should": [
            {"match": {"content": {"query": query, "operator": "or"}}},
            {"match_phrase": {"content": {"query": query, "slop": 3, "boost": 4}}},
        ], "minimum_should_match": 1}}, source=["invoice_id", "filename"])
        semantic = es_client.search(index=INDEX, size=window, knn={
            "field": "embedding", "query_vector": vector,
            "k": window, "num_candidates": max(100, limit * 10),
        }, source=["invoice_id", "filename"])
        hits = reciprocal_rank_fusion(lexical["hits"]["hits"], semantic["hits"]["hits"], limit=limit)
    except Exception as error:
        raise HTTPException(503, f"No se pudo completar la búsqueda: {type(error).__name__}") from error
    return {"results": [
        {**hit["_source"], "score": hit["_score"], "file_url": f"/api/documents/{hit['_source']['invoice_id']}/file"}
        for hit in hits
    ]}


@mcp.tool()
def index_invoice(filename: str, content_base64: str) -> dict:
    """Encola un PDF individual enviado como base64 y devuelve su estado."""
    try:
        if len(content_base64) > 3_600_000:
            return {"status": "rejected", "error": "Para PDFs grandes, use la API REST multipart"}
        data = base64.b64decode(content_base64, validate=True)
        if len(data) > MAX_BYTES:
            return {"status": "rejected", "error": "El PDF supera el límite configurado"}
        data = normalize_pdf(data)
    except (ValueError, binascii.Error) as error:
        return {"status": "rejected", "error": str(error)}
    digest = hashlib.sha256(data).hexdigest()
    with closing(connect()) as db:
        row = db.execute("SELECT id FROM jobs WHERE sha256=?", (digest,)).fetchone()
        if row:
            return {"id": row["id"], "status": "duplicate"}
        doc_id = str(uuid.uuid4())
        object_key = f"documents/{doc_id}.pdf"
        put_object(object_key, data)
        try:
            db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)", (doc_id, digest, filename, object_key, "queued"))
            db.commit()
        except Exception as error:
            db.rollback()
            try:
                delete_object(object_key)
            except Exception:
                pass
            if "unique" not in str(error).lower():
                return {"status": "failed", "error": str(error)[:300]}
            row = db.execute("SELECT id FROM jobs WHERE sha256=?", (digest,)).fetchone()
            return {"id": row["id"], "status": "duplicate"}
    return {"id": doc_id, "status": "queued"}


@mcp.tool()
def get_invoice_status(document_id: str) -> dict:
    """Consulta el estado de extracción e indexación de una factura."""
    with closing(connect()) as db:
        row = db.execute("SELECT id,filename,status,error,created_at FROM jobs WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else {"error": "Documento no encontrado"}


@mcp.tool()
def semantic_search(query: str, limit: int = 10) -> dict:
    """Busca facturas por significado y texto; limit debe estar entre 1 y 50."""
    if not 1 <= limit <= 50:
        return {"error": "limit debe estar entre 1 y 50"}
    try:
        return run_search(query, limit)
    except HTTPException as error:
        return {"error": error.detail}


app.mount("/", mcp.streamable_http_app())


HTML = '''<!doctype html><html lang="es"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>Facturas · búsqueda semántica</title><style>
:root{font:16px/1.5 Inter,ui-sans-serif,system-ui;color:#e8edf5;background:#0b0f14;--panel:#121922;--line:#253140;--muted:#8e9bad;--accent:#a3e635}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 50% -25%,#1d2b2d 0,transparent 48%),#0b0f14}header,.wrap{max-width:1040px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}h1{font-size:1.1rem;margin:0;letter-spacing:.02em}h2{font-size:1.15rem}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;width:34px;height:34px;background:var(--accent);color:#101510;border-radius:10px;font-weight:900}.actions,.tabs{display:flex;gap:8px;align-items:center}button,input{font:inherit;color:inherit}button{border:1px solid var(--line);background:#1a2430;border-radius:9px;padding:10px 15px;cursor:pointer}button:hover{border-color:#71809a}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:var(--accent);color:#101510;border-color:var(--accent);font-weight:700}.wrap{padding-top:48px}.hero{max-width:770px;margin:8vh auto 3rem;text-align:center}.eyebrow{color:var(--accent);font-size:.78rem;text-transform:uppercase;letter-spacing:.15em}.hero h2{font-size:clamp(2rem,5vw,3.5rem);line-height:1.08;margin:14px 0}.sub{color:var(--muted)}.searchbox{display:flex;gap:8px;margin:28px 0 12px;padding:8px;background:var(--panel);border:1px solid var(--line);border-radius:14px}.searchbox input{flex:1;min-width:0;background:transparent;border:0;outline:0;padding:10px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}.tabs{border-bottom:1px solid var(--line);padding-bottom:12px}.tab.active{color:var(--accent);border-color:#51672d}.hidden{display:none!important}input[type=text],input[type=password],input[type=url],input[type=number]{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:#0c1219}.field{display:grid;gap:7px;margin:14px 0}.muted,.hint{color:var(--muted);font-size:.9rem}ul{padding-left:22px}li{padding:8px 0;border-bottom:1px solid var(--line)}a{color:var(--accent)}pre{white-space:pre-wrap;color:#c4d0df;font-size:.85rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.notice{padding:14px;border:1px solid #37482b;border-radius:10px;color:#c6dd9e}@media(max-width:600px){header,.wrap{padding:18px}.wrap{padding-top:28px}.hero{margin-top:6vh}.grid{grid-template-columns:1fr}.searchbox{align-items:stretch}.searchbox button{padding:9px}}
</style><header><div class="brand"><span class="mark">F</span><h1>Facturas</h1></div><div class="actions"><span id="auth-indicator" class="muted">Sin autenticar</span><button id="settings-shortcut">⚙ Ajustes</button></div></header><main class="wrap"><div id="auth-gate" class="panel hidden"><h2>Conecta con tu API</h2><p class="muted">Introduce el token para habilitar búsqueda, documentos y configuración.</p><div class="actions"><input id="token" type="password" autocomplete="current-password" placeholder="Token API"><button class="primary" id="save-token">Conectar</button></div><p id="auth-status" role="status"></p></div><div class="tabs"><button class="tab active" data-view="search-view">⌕ Buscar</button><button class="tab" data-view="documents-view">Documentos</button></div><section id="search-view"><div class="hero"><p class="eyebrow">Búsqueda semántica</p><h2>Encuentra la factura<br>que necesitas.</h2><p class="sub">Describe un servicio, producto o concepto para buscarlo entre tus documentos.</p><form id="search" class="searchbox"><input id="query" required placeholder="Ej. mantenimiento de impresoras"><button class="primary">Buscar</button></form><p class="hint">Resultados por similitud semántica y coincidencia de texto</p></div><section class="panel hidden" id="results-panel"><h2>Resultados</h2><ul id="results"></ul><p class="hint">MarkItDown extrae texto. OCR está desactivado; los vectores no representan la apariencia visual del PDF.</p></section></section><section id="documents-view" class="hidden"><div class="panel"><h2>Subir documentos</h2><p class="muted">Añade uno o varios archivos PDF al índice.</p><div class="field"><label for="files">Archivos PDF</label><input id="files" type="file" accept="application/pdf,.pdf" multiple></div><button class="primary" id="upload">Subir archivos</button><p id="upload-status" role="status" aria-live="polite"></p></div><div class="panel"><div class="actions" style="justify-content:space-between"><h2>Biblioteca</h2><span id="page" class="muted"></span></div><ul id="docs"></ul><button id="previous">Anterior</button> <button id="next">Siguiente</button></div></section><section id="settings-view" class="hidden"><div class="panel"><h2>Configuración</h2><p class="muted">Las opciones quedan disponibles al autenticarte con el token API.</p><h3>Embeddings</h3><form id="embedding-form"><div class="grid"><label class="field">URL del servicio<input id="embedding-url" type="url" required></label><div class="field"><label for="embedding-model">Modelo</label><div class="actions"><input id="embedding-model" type="text" required list="embedding-models"><datalist id="embedding-models"></datalist><button type="button" id="detect-models">Detectar</button></div></div></div><label class="field">Clave del proveedor<input id="embedding-key" type="password" placeholder="Dejar vacío para conservar la configurada"></label><button class="primary">Guardar configuración</button><span id="settings-status" role="status" class="muted"></span></form></div><div class="panel"><h3>Conexión MCP</h3><p class="muted">Transporte: Streamable HTTP · Endpoint: <code>/mcp</code> · Autenticación: Bearer token API</p><p class="muted">Herramientas: <code>index_invoice</code>, <code>get_invoice_status</code>, <code>semantic_search</code></p><h3>Estado del servicio</h3><pre id="health">Consultando…</pre></div></section></main><script>
const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n};let offset=0,total=0;const pageSize=50;const token=document.querySelector('#token');token.value=sessionStorage.getItem('api-token')||'';const hasToken=()=>Boolean(sessionStorage.getItem('api-token'));const gate=document.querySelector('#auth-gate');
async function api(url,options={}){options.headers={...options.headers,Authorization:`Bearer ${sessionStorage.getItem('api-token')||''}`};return fetch(url,options)}
function authUI(ok){gate.classList.toggle('hidden',ok);document.querySelector('#auth-indicator').textContent=ok?'Conectado':'Sin autenticar';document.querySelectorAll('.tab').forEach(b=>b.disabled=!ok);document.querySelector('#settings-shortcut').disabled=!ok;document.querySelector('#query').disabled=!ok;document.querySelector('#search button').disabled=!ok;document.querySelector('#upload').disabled=!ok}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));['search-view','documents-view','settings-view'].forEach(id=>document.querySelector(`#${id}`).classList.toggle('hidden',id!==b.dataset.view))});
document.querySelector('#settings-shortcut').onclick=()=>{document.querySelector('[data-view="search-view"]').classList.remove('active');document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));['search-view','documents-view'].forEach(id=>document.querySelector(`#${id}`).classList.add('hidden'));document.querySelector('#settings-view').classList.remove('hidden');loadSettings()};
document.querySelector('#save-token').onclick=async()=>{const value=token.value.trim();if(!value)return;sessionStorage.setItem('api-token',value);try{const response=await api('/api/settings/embeddings');if(!response.ok)throw new Error('Token no válido');authUI(true);document.querySelector('#auth-status').textContent='';loadSettings();refresh()}catch{sessionStorage.removeItem('api-token');authUI(false);document.querySelector('#auth-status').textContent='No se pudo validar el token.'}};
async function loadSettings(){if(!hasToken())return;const [s,h]=await Promise.all([api('/api/settings/embeddings'),fetch('/api/health')]);if(s.ok){const d=await s.json();document.querySelector('#embedding-url').value=d.base_url;document.querySelector('#embedding-model').value=d.model}document.querySelector('#health').textContent=JSON.stringify(await h.json(),null,2)}
document.querySelector('#detect-models').onclick=async()=>{const status=document.querySelector('#settings-status');status.textContent='Detectando modelos…';try{const r=await api('/api/settings/embeddings/models');const d=await r.json();if(!r.ok)throw Error(d.detail);const list=document.querySelector('#embedding-models');list.replaceChildren(...d.models.map(id=>{const option=document.createElement('option');option.value=id;return option}));status.textContent=`${d.models.length} modelos detectados. Selecciónalos o escribe un ID manualmente.`}catch(error){status.textContent=error.message}};
document.querySelector('#embedding-form').onsubmit=async e=>{e.preventDefault();const status=document.querySelector('#settings-status');const r=await api('/api/settings/embeddings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_url:document.querySelector('#embedding-url').value,model:document.querySelector('#embedding-model').value,api_key:document.querySelector('#embedding-key').value})});status.textContent=r.ok?'Configuración guardada.':'No se pudo guardar la configuración.';document.querySelector('#embedding-key').value='';loadSettings()};
async function download(url,name){const r=await api(url);if(!r.ok)return;const link=document.createElement('a');link.href=URL.createObjectURL(await r.blob());link.download=name;link.click();URL.revokeObjectURL(link.href)}
async function refresh(){if(!hasToken())return;const response=await api(`/api/documents?limit=${pageSize}&offset=${offset}`);if(!response.ok)return;const d=await response.json();total=d.total;const list=document.querySelector('#docs');list.replaceChildren();for(const x of d.documents){const li=el('li','');const a=el('a',x.filename);a.href=`/api/documents/${x.id}/file`;a.onclick=e=>{e.preventDefault();download(a.href,x.filename)};li.append(a,document.createTextNode(` · ${x.status}${x.error?` · ${x.error}`:''} `));if(x.status==='failed'){const b=el('button','Reintentar');b.onclick=()=>api(`/api/documents/${x.id}/retry`,{method:'POST'}).then(refresh);li.append(b)}list.append(li)}document.querySelector('#page').textContent=`${total} documentos · ${total?offset+1:0}-${Math.min(offset+pageSize,total)}`;document.querySelector('#previous').disabled=offset===0;document.querySelector('#next').disabled=offset+pageSize>=total}
document.querySelector('#previous').onclick=()=>{offset=Math.max(0,offset-pageSize);refresh()};document.querySelector('#next').onclick=()=>{offset+=pageSize;refresh()};document.querySelector('#upload').onclick=async()=>{const files=document.querySelector('#files').files;const status=document.querySelector('#upload-status');if(!files.length){status.textContent='Selecciona al menos un PDF.';return}const body=new FormData();for(const f of files)body.append('files',f);status.textContent=`Enviando ${files.length} archivo(s)…`;try{const r=await api('/api/documents',{method:'POST',body});const d=await r.json();if(!r.ok)throw Error(d.detail||'Error al subir');status.textContent=`${d.accepted} en cola, ${d.duplicates} duplicados, ${d.rejected} rechazados.`;refresh()}catch(e){status.textContent=`No se pudo subir: ${e.message}`}};
document.querySelector('#search').onsubmit=async e=>{e.preventDefault();const list=document.querySelector('#results');const panel=document.querySelector('#results-panel');panel.classList.remove('hidden');list.replaceChildren(el('li','Buscando…'));try{const r=await api('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.querySelector('#query').value})});const d=await r.json();list.replaceChildren();if(!r.ok)throw Error(d.detail||'Error en búsqueda');if(!d.results.length)list.append(el('li','No se encontraron facturas.'));for(const x of d.results){const li=el('li','');const a=el('a',x.filename);a.href=x.file_url;a.onclick=ev=>{ev.preventDefault();download(a.href,x.filename)};li.append(a,document.createTextNode(` · relevancia ${x.score.toFixed(4)}`));list.append(li)}}catch(error){list.replaceChildren(el('li',error.message))}};
authUI(false);if(hasToken())document.querySelector('#save-token').click();setInterval(refresh,5000);
</script></html>'''
