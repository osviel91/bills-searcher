import hashlib
import hmac
import json
import mimetypes
import os
import tempfile
import threading
import time
import uuid
from io import BytesIO
from contextlib import asynccontextmanager, closing
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from application.search import (
    INVOICE_SEARCH_FIELDS,
    ServiceUnavailable,
    reciprocal_rank_fusion,
    run_invoice_metadata as load_invoice_metadata,
    run_invoice_evidence_search as search_invoice_evidence_service,
    run_invoice_filter as search_invoice_filter,
    run_search as search_invoices,
)
from infrastructure.embeddings import MODEL, available_models, health as embedding_health, settings as embedding_settings, update_settings
from domain.evidence_evaluation import invoice_recall_at_k, recall_at_k, scoped_recall_at_k
from infrastructure.jobstore import connect as connect_jobs, processing_metrics, visible_generation_query
from infrastructure.object_store import container_client, delete as delete_object, ensure_container, get as get_object, put as put_object
from infrastructure.storage import INDEX, PASSAGES_INDEX, client as es, initialize_index
from interfaces.mcp import mcp, register_tools

DATA = Path(os.getenv("DATA_DIR", "/data"))
DB = DATA / "jobs.sqlite3"
EVIDENCE_EVALUATION_LOCK = threading.Lock()
INDEX_INITIALIZATION_ERROR = None
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


@asynccontextmanager
async def lifespan(_app):
    global INDEX_INITIALIZATION_ERROR
    with closing(connect()):
        pass
    ensure_container()
    try:
        initialize_index(es())
        INDEX_INITIALIZATION_ERROR = None
    except Exception as error:
        INDEX_INITIALIZATION_ERROR = type(error).__name__
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
        "son `index_invoice`, `get_invoice_status`, `reindex_invoices`, `semantic_search`, "
        "`search_invoice_evidence` y `filter_invoices`."
    ),
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {"name": "Facturas", "description": "Carga por bloques, consulta de estado, reintento y descarga de PDFs."},
        {"name": "Búsqueda", "description": "Búsqueda híbrida léxica y semántica sobre facturas indexadas."},
        {"name": "Evaluación", "description": "Etiquetado local de evidencia y medición de Recall@K."},
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


def normalize_evaluation_cases(request: "EvidenceEvaluationSet", *, require_labels=False):
    cases = [case.model_dump() for case in request.cases]
    if len({case["id"] for case in cases}) != len(cases):
        raise HTTPException(422, "Cada caso de evaluación debe tener un id único")
    for case in cases:
        if not case["query"].strip():
            raise HTTPException(422, "La consulta de cada caso no puede estar vacía")
        if require_labels and not case["relevant_evidence"]:
            raise HTTPException(422, "Cada caso debe tener al menos una cita etiquetada")
        if any(not item["invoice_id"].strip() or not item["quote"].strip()
               for item in case["relevant_evidence"]):
            raise HTTPException(422, "Cada cita debe tener invoice_id y texto no vacíos")
    return cases


def load_evidence_evaluation_cases():
    path = DATA / "evidence-evaluation.json"
    with EVIDENCE_EVALUATION_LOCK:
        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(500, "No se pudo leer el conjunto local de evaluación") from error
    try:
        return EvidenceEvaluationSet(cases=cases).model_dump()["cases"]
    except Exception as error:
        raise HTTPException(500, "El conjunto local de evaluación no es válido") from error


def save_evidence_evaluation_cases(cases):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "evidence-evaluation.json"
    with EVIDENCE_EVALUATION_LOCK:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=DATA,
                                             prefix=".evidence-evaluation-", suffix=".tmp",
                                             delete=False) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(cases, temporary, ensure_ascii=False, indent=2)
            os.chmod(temporary_path, 0o600)
            temporary_path.replace(path)
        except OSError as error:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)
            raise HTTPException(500, "No se pudo guardar el conjunto local de evaluación") from error


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Concepto o descripción que se quiere encontrar en las facturas.", examples=["mantenimiento de impresoras"])
    limit: int = Field(default=10, ge=1, le=50, description="Máximo de resultados que se devolverán.")


class EvidenceSearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Texto para recuperar pasajes de evidencia.")
    limit: int = Field(default=5, ge=1, le=20)
    invoice_id: str | None = None


class InvoiceFilterRequest(BaseModel):
    customer_code: str | None = None
    customer_tax_id: str | None = None
    invoice_number: str | None = None
    delivery_note_id: str | None = None
    concept: str | None = None
    currency: str | None = None
    invoice_date_from: date | None = None
    invoice_date_to: date | None = None
    total_payable_min: float | None = None
    total_payable_max: float | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class ExpectedEvidence(BaseModel):
    invoice_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=2000)


class EvidenceEvaluationCase(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    relevant_evidence: list[ExpectedEvidence] = Field(default_factory=list, max_length=20)


class EvidenceEvaluationSet(BaseModel):
    cases: list[EvidenceEvaluationCase] = Field(default_factory=list, max_length=30)


class EmbeddingSettings(BaseModel):
    provider: Literal["openai-compatible", "jina-local"] = "openai-compatible"
    document_mode: Literal["text", "direct"] = "text"
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
    invoice_number: str | None = None
    invoice_date: str | None = None
    customer_code: str | None = None
    customer_tax_id: str | None = None
    delivery_note_ids: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    currency: str | None = None
    total_payable: float | None = None
    invoice_total: float | None = None
    tax_base: float | None = None
    vat: float | None = None
    confidence: dict = Field(default_factory=dict)
    evidence: dict = Field(default_factory=dict)


class DocumentMetadataResponse(BaseModel):
    invoice_id: str
    filename: str
    status: Literal["indexed"]
    file_url: str
    index_record_available: bool
    metadata_available: bool
    metadata: dict
    confidence: dict = Field(default_factory=dict)
    evidence: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult]


class HealthResponse(BaseModel):
    elasticsearch: bool
    index_initialization_error: str | None = None
    database: bool
    object_storage: bool
    embedding_service: bool
    configured_model: str
    model_available: bool
    document_processing: str
    pdf_representation: str
    metrics: dict = Field(default_factory=dict)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/health", tags=["Sistema"], summary="Comprobar el estado de las dependencias",
         response_model=HealthResponse)
def health(response: Response):
    try:
        es_ok = es().ping() and INDEX_INITIALIZATION_ERROR is None
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
    try:
        metrics = processing_metrics(DB)
    except Exception:
        metrics = {}
        database_ok = False
    if not (es_ok and database_ok and object_store_ok):
        response.status_code = 503
    return {"elasticsearch": es_ok, "index_initialization_error": INDEX_INITIALIZATION_ERROR,
            "database": database_ok, "object_storage": object_store_ok,
            "embedding_service": inference_ok,
            "configured_model": embedding_settings()["model"],
            "model_available": model_available,
            "document_processing": ("Jina direct PDF/image embeddings" if embedding_settings()["document_mode"] == "direct"
                                    else "MarkItDown text extraction; OCR disabled; scanned PDFs may produce no text"),
            "pdf_representation": ("multimodal PDF/image embeddings" if embedding_settings()["document_mode"] == "direct"
                                   else "text-derived embeddings, not multimodal PDF embeddings"),
            "metrics": metrics}


@app.get("/api/health/live", tags=["Sistema"], include_in_schema=False)
def liveness():
    return {"status": "live"}


@app.get("/api/health/ready", tags=["Sistema"], include_in_schema=False)
def readiness(response: Response):
    state = health(response)
    return {"ready": response.status_code == 200, "dependencies": {
        "database": state["database"], "elasticsearch": state["elasticsearch"],
        "object_storage": state["object_storage"],
    }}


@app.get("/api/settings/embeddings", dependencies=[Depends(require_token)])
def get_embedding_settings():
    config = embedding_settings()
    return {"provider": config["provider"], "base_url": config["base_url"],
            "model": config["model"], "document_mode": config["document_mode"],
            "dimensions": config["dimensions"], "api_key_configured": bool(config["api_key"])}


@app.put("/api/settings/embeddings", dependencies=[Depends(require_token)])
def put_embedding_settings(config: EmbeddingSettings):
    current = embedding_settings()
    if config.document_mode == "direct" and config.provider != "jina-local":
        raise HTTPException(422, "El modo directo PDF/imagen requiere Jina local")
    update_settings(config.base_url, config.model, config.api_key or current["api_key"],
                    config.provider, config.document_mode)
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
    summary="Subir PDFs e imágenes",
    description=("Envía archivos en el campo multipart repetido `files`. Imágenes y PDFs directos requieren "
                 "Jina local en modo directo; el modo texto acepta PDFs con texto extraíble."),
    response_model=UploadResponse,
    responses={
        401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
        413: {"model": ApiError, "description": "El bloque supera el número máximo de archivos."},
        503: {"model": ApiError, "description": "El token no está configurado."},
    },
)
async def upload(files: list[UploadFile] = File(..., description="Uno o más PDFs o imágenes; repite `files` por archivo.")):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"El bloque supera el límite de {MAX_BATCH_FILES} ficheros")
    embedding_config = embedding_settings()
    created = []
    with closing(connect()) as db:
        for file in files:
            try:
                suffix = Path(file.filename or "").suffix.lower()
                content_type = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix)
                if not content_type:
                    raise ValueError("Solo se admiten PDF, PNG, JPEG y WEBP")
                if embedding_config["document_mode"] == "direct" and embedding_config["provider"] != "jina-local":
                    raise ValueError("El modo directo PDF/imagen requiere Jina local")
                data = await file.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    created.append({"filename": file.filename, "status": "rejected", "error": "El archivo supera el límite configurado"})
                    continue
                if content_type == "application/pdf":
                    data = normalize_pdf(data)
                else:
                    if embedding_config["provider"] != "jina-local" or embedding_config["document_mode"] != "direct":
                        raise ValueError("Las imágenes requieren Jina local en modo directo")
                    try:
                        with Image.open(BytesIO(data)) as image:
                            if image.format not in {"PNG", "JPEG", "WEBP"}:
                                raise ValueError("Formato de imagen no admitido")
                            image.verify()
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        raise ValueError("La imagen no es un PNG, JPEG o WEBP válido") from error
            except ValueError as error:
                created.append({"filename": file.filename, "status": "rejected", "error": str(error)})
                continue
            digest = hashlib.sha256(data).hexdigest()
            row = db.execute("SELECT id FROM jobs WHERE sha256=?", (digest,)).fetchone()
            if row:
                created.append({"id": row["id"], "filename": file.filename, "status": "duplicate"})
                continue
            doc_id = str(uuid.uuid4())
            object_key = f"documents/{doc_id}{suffix}"
            stored = False
            try:
                await run_in_threadpool(put_object, object_key, data, content_type)
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


@app.get("/api/documents/{doc_id}/metadata", dependencies=[Depends(require_token)], tags=["Facturas"],
         summary="Consultar los metadatos genéricos indexados de una factura",
         response_model=DocumentMetadataResponse,
         responses={401: {"model": ApiError, "description": "Bearer token ausente o no válido."},
                    404: {"model": ApiError, "description": "No existe una factura con ese ID."},
                    409: {"model": ApiError, "description": "La factura todavía no está indexada."},
                    503: {"model": ApiError, "description": "El índice de facturas no está disponible."}})
def document_metadata(doc_id: str):
    try:
        return load_invoice_metadata(doc_id)
    except LookupError as error:
        raise HTTPException(404, "Documento no encontrado") from error
    except ValueError as error:
        raise HTTPException(409, "Documento no indexado") from error
    except ServiceUnavailable as error:
        raise HTTPException(503, str(error)) from error


@app.get("/api/documents/{doc_id}/text", dependencies=[Depends(require_token)], tags=["Facturas"],
         summary="Leer el texto extraído para revisar evidencia")
def document_text(doc_id: str):
    try:
        with closing(connect()) as db:
            row = db.execute("SELECT active_generation_id,status FROM jobs WHERE id=?", (doc_id,)).fetchone()
        if not row or (row["status"] != "indexed" and not row["active_generation_id"]):
            raise HTTPException(404, "Texto extraído no encontrado")
        response = es().search(index=INDEX, size=1, query={"bool": {"filter": [
            {"term": {"invoice_id": doc_id}}, visible_generation_query(path=DB),
        ]}},
                               source=["invoice_id", "filename", "content"])
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(503, f"No se pudo leer el texto extraído: {type(error).__name__}") from error
    hits = response["hits"]["hits"]
    if not hits:
        raise HTTPException(404, "Texto extraído no encontrado")
    return {**hits[0]["_source"], "file_url": f"/api/documents/{doc_id}/file"}


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
        result = db.execute("""UPDATE jobs SET status='queued',error=NULL,
                             active_generation_id=COALESCE(active_generation_id,'legacy')
                             WHERE id=? AND status='failed'""", (doc_id,))
        db.commit()
        if not result.rowcount:
            raise HTTPException(404, "Documento fallido no encontrado")
    return {"id": doc_id, "status": "queued"}


@app.get(
    "/api/documents/{doc_id}/file",
    dependencies=[Depends(require_token)],
    tags=["Facturas"],
    summary="Descargar el archivo original",
    responses={
        200: {"description": "PDF o imagen original.", "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}, "image/*": {"schema": {"type": "string", "format": "binary"}}}},
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

    return StreamingResponse(chunks(), media_type=mimetypes.guess_type(row["filename"])[0] or "application/octet-stream",
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
        return search_invoices(query, limit)
    except ServiceUnavailable as error:
        raise HTTPException(503, str(error)) from error


def run_invoice_filter(*, customer_code=None, customer_tax_id=None, invoice_number=None,
                       delivery_note_id=None, concept=None, currency=None, invoice_date_from=None,
                       invoice_date_to=None, total_payable_min=None, total_payable_max=None,
                       limit=50, offset=0):
    try:
        return search_invoice_filter(
            customer_code=customer_code, customer_tax_id=customer_tax_id,
            invoice_number=invoice_number, delivery_note_id=delivery_note_id,
            concept=concept, currency=currency, invoice_date_from=invoice_date_from,
            invoice_date_to=invoice_date_to, total_payable_min=total_payable_min,
            total_payable_max=total_payable_max, limit=limit, offset=offset,
        )
    except ServiceUnavailable as error:
        raise HTTPException(503, str(error)) from error


def run_invoice_evidence_search(query: str, limit: int = 5, invoice_id: str | None = None):
    try:
        return search_invoice_evidence_service(query, limit, invoice_id)
    except ServiceUnavailable as error:
        raise HTTPException(503, str(error)) from error


@app.post("/api/search/evidence", dependencies=[Depends(require_token)], tags=["Búsqueda"],
          summary="Recuperar pasajes de evidencia", responses={401: {"model": ApiError}, 503: {"model": ApiError}})
def search_evidence(request: EvidenceSearchRequest):
    if not request.query.strip():
        raise HTTPException(422, "query no puede estar vacío")
    return run_invoice_evidence_search(request.query, request.limit, request.invoice_id)


@app.post("/api/invoices/filter", dependencies=[Depends(require_token)], tags=["Búsqueda"],
          summary="Filtrar facturas por campos extraídos", responses={401: {"model": ApiError}, 503: {"model": ApiError}})
def filter_invoices_rest(request: InvoiceFilterRequest):
    if request.invoice_date_from and request.invoice_date_to and request.invoice_date_from > request.invoice_date_to:
        raise HTTPException(422, "invoice_date_from no puede ser posterior a invoice_date_to")
    if (request.total_payable_min is not None and request.total_payable_max is not None
            and request.total_payable_min > request.total_payable_max):
        raise HTTPException(422, "total_payable_min no puede superar total_payable_max")
    return run_invoice_filter(
        customer_code=request.customer_code, customer_tax_id=request.customer_tax_id,
        invoice_number=request.invoice_number, delivery_note_id=request.delivery_note_id,
        concept=request.concept, currency=request.currency,
        invoice_date_from=request.invoice_date_from.isoformat() if request.invoice_date_from else None,
        invoice_date_to=request.invoice_date_to.isoformat() if request.invoice_date_to else None,
        total_payable_min=request.total_payable_min, total_payable_max=request.total_payable_max,
        limit=request.limit, offset=request.offset,
    )


@app.get("/api/evaluation/evidence/cases", dependencies=[Depends(require_token)], tags=["Evaluación"],
         summary="Leer las etiquetas locales de evidencia")
def get_evidence_evaluation_cases():
    return {"cases": load_evidence_evaluation_cases()}


@app.put("/api/evaluation/evidence/cases", dependencies=[Depends(require_token)], tags=["Evaluación"],
         summary="Guardar etiquetas locales de evidencia")
def put_evidence_evaluation_cases(request: EvidenceEvaluationSet):
    cases = normalize_evaluation_cases(request)
    save_evidence_evaluation_cases(cases)
    return {"cases": cases, "saved": len(cases)}


@app.post("/api/evaluation/evidence/run", dependencies=[Depends(require_token)], tags=["Evaluación"],
          summary="Calcular Recall@5/10 de las etiquetas")
def run_evidence_evaluation(request: EvidenceEvaluationSet):
    cases = normalize_evaluation_cases(request, require_labels=True)
    if not cases:
        raise HTTPException(422, "Añade al menos un caso de evaluación")
    scores = []
    results = []
    for case in cases:
        started = time.perf_counter()
        invoice_results = run_search(case["query"], limit=10)["results"]
        global_evidence = run_invoice_evidence_search(case["query"], limit=10)["evidence"]
        relevant = case["relevant_evidence"]
        relevant_invoice_ids = {item["invoice_id"] for item in relevant}
        scoped_evidence = {
            invoice_id: run_invoice_evidence_search(case["query"], limit=10,
                                                     invoice_id=invoice_id)["evidence"]
            for invoice_id in sorted(relevant_invoice_ids)
        }
        metrics = (
            invoice_recall_at_k(relevant_invoice_ids, invoice_results, 5),
            invoice_recall_at_k(relevant_invoice_ids, invoice_results, 10),
            recall_at_k(relevant, global_evidence, 5),
            recall_at_k(relevant, global_evidence, 10),
            scoped_recall_at_k(relevant, scoped_evidence, 5),
            scoped_recall_at_k(relevant, scoped_evidence, 10),
        )
        scores.append(metrics)
        results.append({
            "id": case["id"], "query": case["query"],
            "relevant_evidence_count": len(relevant),
            "invoice_recall_at_5": metrics[0], "invoice_recall_at_10": metrics[1],
            "global_evidence_recall_at_5": metrics[2], "global_evidence_recall_at_10": metrics[3],
            "scoped_evidence_recall_at_5": metrics[4], "scoped_evidence_recall_at_10": metrics[5],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "invoice_top10": invoice_results[:10],
            "global_evidence_top10": global_evidence[:10],
            "scoped_evidence_top10": [
                {"invoice_id": invoice_id, "evidence": hits}
                for invoice_id, hits in scoped_evidence.items()
            ],
        })
    count = len(scores)
    return {
        "results": results,
        "summary": {
            "cases": count,
            "mean_invoice_recall_at_5": sum(score[0] for score in scores) / count,
            "mean_invoice_recall_at_10": sum(score[1] for score in scores) / count,
            "mean_global_evidence_recall_at_5": sum(score[2] for score in scores) / count,
            "mean_global_evidence_recall_at_10": sum(score[3] for score in scores) / count,
            "mean_scoped_evidence_recall_at_5": sum(score[4] for score in scores) / count,
            "mean_scoped_evidence_recall_at_10": sum(score[5] for score in scores) / count,
        },
    }


MCP_TOOLS = register_tools(
    max_bytes=lambda: globals()["MAX_BYTES"],
    connect=lambda: globals()["connect"](),
    normalize_pdf=lambda *args, **kwargs: globals()["normalize_pdf"](*args, **kwargs),
    put_object=lambda *args, **kwargs: globals()["put_object"](*args, **kwargs),
    delete_object=lambda *args, **kwargs: globals()["delete_object"](*args, **kwargs),
    run_search=lambda *args, **kwargs: globals()["run_search"](*args, **kwargs),
    run_invoice_filter=lambda *args, **kwargs: globals()["run_invoice_filter"](*args, **kwargs),
    run_invoice_evidence_search=lambda *args, **kwargs: globals()["run_invoice_evidence_search"](*args, **kwargs),
)
globals().update(MCP_TOOLS)
app.mount("/", mcp.streamable_http_app())


HTML = '''<!doctype html><html lang="es"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>Facturas · búsqueda semántica</title><style>
:root{font:16px/1.5 Inter,ui-sans-serif,system-ui;color:#e8edf5;background:#0b0f14;--panel:#121922;--line:#253140;--muted:#8e9bad;--accent:#a3e635}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 50% -25%,#1d2b2d 0,transparent 48%),#0b0f14}header,.wrap{max-width:1040px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}h1{font-size:1.1rem;margin:0;letter-spacing:.02em}h2{font-size:1.15rem}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;width:34px;height:34px;background:var(--accent);color:#101510;border-radius:10px;font-weight:900}.actions,.tabs{display:flex;gap:8px;align-items:center}button,input{font:inherit;color:inherit}button{border:1px solid var(--line);background:#1a2430;border-radius:9px;padding:10px 15px;cursor:pointer}button:hover{border-color:#71809a}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:var(--accent);color:#101510;border-color:var(--accent);font-weight:700}.wrap{padding-top:48px}.hero{max-width:770px;margin:8vh auto 3rem;text-align:center}.eyebrow{color:var(--accent);font-size:.78rem;text-transform:uppercase;letter-spacing:.15em}.hero h2{font-size:clamp(2rem,5vw,3.5rem);line-height:1.08;margin:14px 0}.sub{color:var(--muted)}.searchbox{display:flex;gap:8px;margin:28px 0 12px;padding:8px;background:var(--panel);border:1px solid var(--line);border-radius:14px}.searchbox input{flex:1;min-width:0;background:transparent;border:0;outline:0;padding:10px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}.tabs{border-bottom:1px solid var(--line);padding-bottom:12px}.tab.active{color:var(--accent);border-color:#51672d}.hidden{display:none!important}input[type=text],input[type=password],input[type=url],input[type=number]{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:#0c1219}.field{display:grid;gap:7px;margin:14px 0}.muted,.hint{color:var(--muted);font-size:.9rem}ul{padding-left:22px}li{padding:8px 0;border-bottom:1px solid var(--line)}a{color:var(--accent)}pre{white-space:pre-wrap;color:#c4d0df;font-size:.85rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.notice{padding:14px;border:1px solid #37482b;border-radius:10px;color:#c6dd9e}@media(max-width:600px){header,.wrap{padding:18px}.wrap{padding-top:28px}.hero{margin-top:6vh}.grid{grid-template-columns:1fr}.searchbox{align-items:stretch}.searchbox button{padding:9px}}
</style><header><div class="brand"><span class="mark">F</span><h1>Facturas</h1></div><div class="actions"><span id="auth-indicator" class="muted">Sin autenticar</span><button id="settings-shortcut">⚙ Ajustes</button></div></header><main class="wrap"><div id="auth-gate" class="panel hidden"><h2>Conecta con tu API</h2><p class="muted">Introduce el token para habilitar búsqueda, documentos y configuración.</p><div class="actions"><input id="token" type="password" autocomplete="current-password" placeholder="Token API"><button class="primary" id="save-token">Conectar</button></div><p id="auth-status" role="status"></p></div><div class="tabs"><button class="tab active" data-view="search-view">⌕ Buscar</button><button class="tab" data-view="documents-view">Documentos</button></div><section id="search-view"><div class="hero"><p class="eyebrow">Búsqueda semántica</p><h2>Encuentra la factura<br>que necesitas.</h2><p class="sub">Describe un servicio, producto o concepto para buscarlo entre tus documentos.</p><form id="search" class="searchbox"><input id="query" required placeholder="Ej. mantenimiento de impresoras"><button class="primary">Buscar</button></form><p class="hint">Resultados por similitud semántica y coincidencia de texto</p></div><section class="panel hidden" id="results-panel"><h2>Resultados</h2><ul id="results"></ul><p class="hint">MarkItDown extrae texto. OCR está desactivado; los vectores no representan la apariencia visual del PDF.</p></section></section><section id="documents-view" class="hidden"><div class="panel"><h2>Subir documentos</h2><p class="muted">Añade uno o varios archivos PDF al índice.</p><div class="field"><label for="files">Archivos PDF</label><input id="files" type="file" accept="application/pdf,.pdf" multiple></div><button class="primary" id="upload">Subir archivos</button><p id="upload-status" role="status" aria-live="polite"></p></div><div class="panel"><div class="actions" style="justify-content:space-between"><h2>Biblioteca</h2><span id="page" class="muted"></span></div><ul id="docs"></ul><button id="previous">Anterior</button> <button id="next">Siguiente</button></div></section><section id="settings-view" class="hidden"><div class="panel"><h2>Configuración</h2><p class="muted">Las opciones quedan disponibles al autenticarte con el token API.</p><h3>Embeddings</h3><form id="embedding-form"><div class="grid"><label class="field">URL del servicio<input id="embedding-url" type="url" required></label><div class="field"><label for="embedding-model">Modelo</label><div class="actions"><input id="embedding-model" type="text" required list="embedding-models"><datalist id="embedding-models"></datalist><button type="button" id="detect-models">Detectar</button></div></div></div><label class="field">Clave del proveedor<input id="embedding-key" type="password" placeholder="Dejar vacío para conservar la configurada"></label><button class="primary">Guardar configuración</button><span id="settings-status" role="status" class="muted"></span></form></div><div class="panel"><h3>Conexión MCP</h3><p class="muted">Transporte: Streamable HTTP · Endpoint: <code>/mcp</code> · Autenticación: Bearer token API</p><p class="muted">Herramientas: <code>index_invoice</code>, <code>get_invoice_status</code>, <code>semantic_search</code></p><h3>Estado del servicio</h3><pre id="health">Consultando…</pre></div></section></main><script>
const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n};let offset=0,total=0;const pageSize=50;const token=document.querySelector('#token');token.value=sessionStorage.getItem('api-token')||'';const hasToken=()=>Boolean(sessionStorage.getItem('api-token'));const gate=document.querySelector('#auth-gate');
const styledSelect=select=>{select.style.cssText='width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:#0c1219;color:inherit'};const providerField=el('div','');providerField.className='field';const providerLabel=el('label','Proveedor de embeddings');providerLabel.htmlFor='embedding-provider';const providerSelect=document.createElement('select');providerSelect.id='embedding-provider';styledSelect(providerSelect);for(const [value,label] of [['openai-compatible','OpenAI-compatible'],['jina-local','Jina local']]){const option=el('option',label);option.value=value;providerSelect.append(option)}providerField.append(providerLabel,providerSelect,el('small','Cambiar proveedor/modelo o dimensiones requiere un índice y reindexado nuevos.'));const modeField=el('div','');modeField.className='field';const modeLabel=el('label','Modo de documentos');modeLabel.htmlFor='embedding-document-mode';const modeSelect=document.createElement('select');modeSelect.id='embedding-document-mode';styledSelect(modeSelect);for(const [value,label] of [['text','Extraer texto del PDF'],['direct','Jina multimodal directo (PDF/imagen)']]){const option=el('option',label);option.value=value;modeSelect.append(option)}const fileInput=document.querySelector('#files');const fileLabel=fileInput.parentElement.querySelector('label');const uploadHelp=document.querySelector('#documents-view .panel p');const resultsHint=document.querySelector('#results-panel .hint');const syncMode=()=>{modeSelect.options[1].disabled=providerSelect.value!=='jina-local';if(modeSelect.value==='direct'&&modeSelect.options[1].disabled)modeSelect.value='text';const direct=modeSelect.value==='direct';fileInput.accept=direct?'application/pdf,.pdf,image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp':'application/pdf,.pdf';fileLabel.textContent=direct?'PDF o imagen':'Archivo PDF';uploadHelp.textContent=direct?'Añade PDFs o imágenes PNG/JPEG/WEBP.':'Añade PDFs con texto extraíble.';resultsHint.textContent=direct?'Jina genera embeddings visuales; el texto extraído también se usa para coincidencias léxicas.':'MarkItDown extrae texto. OCR está desactivado; los vectores no representan la apariencia visual del PDF.'};modeField.append(modeLabel,modeSelect,el('small','Las imágenes solo se admiten en modo directo.'));providerSelect.onchange=syncMode;modeSelect.onchange=syncMode;document.querySelector('#embedding-form').prepend(modeField,providerField);syncMode();
async function api(url,options={}){options.headers={...options.headers,Authorization:`Bearer ${sessionStorage.getItem('api-token')||''}`};return fetch(url,options)}
function authUI(ok){gate.classList.toggle('hidden',ok);document.querySelector('#auth-indicator').textContent=ok?'Conectado':'Sin autenticar';document.querySelectorAll('.tab').forEach(b=>b.disabled=!ok);document.querySelector('#settings-shortcut').disabled=!ok;document.querySelector('#query').disabled=!ok;document.querySelector('#search button').disabled=!ok;document.querySelector('#upload').disabled=!ok}
 document.querySelectorAll('main.wrap > .tabs .tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('main.wrap > .tabs .tab').forEach(x=>x.classList.toggle('active',x===b));['search-view','documents-view','settings-view','evaluation-view'].forEach(id=>document.querySelector(`#${id}`).classList.toggle('hidden',id!==b.dataset.view))});
 document.querySelector('#settings-shortcut').onclick=()=>{document.querySelectorAll('main.wrap > .tabs .tab').forEach(x=>x.classList.remove('active'));['search-view','documents-view','evaluation-view'].forEach(id=>document.querySelector(`#${id}`).classList.add('hidden'));document.querySelector('#settings-view').classList.remove('hidden');loadSettings()};
document.querySelector('#save-token').onclick=async()=>{const value=token.value.trim();if(!value)return;sessionStorage.setItem('api-token',value);try{const response=await api('/api/settings/embeddings');if(!response.ok)throw new Error('Token no válido');authUI(true);document.querySelector('#auth-status').textContent='';loadSettings();refresh()}catch{sessionStorage.removeItem('api-token');authUI(false);document.querySelector('#auth-status').textContent='No se pudo validar el token.'}};
async function loadSettings(){if(!hasToken())return;const [s,h]=await Promise.all([api('/api/settings/embeddings'),fetch('/api/health')]);if(s.ok){const d=await s.json();document.querySelector('#embedding-provider').value=d.provider;syncMode();document.querySelector('#embedding-document-mode').value=d.document_mode;document.querySelector('#embedding-url').value=d.base_url;document.querySelector('#embedding-model').value=d.model}document.querySelector('#health').textContent=JSON.stringify(await h.json(),null,2)}
document.querySelector('#detect-models').onclick=async()=>{const status=document.querySelector('#settings-status');status.textContent='Detectando modelos…';try{const r=await api('/api/settings/embeddings/models');const d=await r.json();if(!r.ok)throw Error(d.detail);const list=document.querySelector('#embedding-models');list.replaceChildren(...d.models.map(id=>{const option=document.createElement('option');option.value=id;return option}));status.textContent=`${d.models.length} modelos detectados. Selecciónalos o escribe un ID manualmente.`}catch(error){status.textContent=error.message}};
document.querySelector('#embedding-form').onsubmit=async e=>{e.preventDefault();const status=document.querySelector('#settings-status');const r=await api('/api/settings/embeddings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:document.querySelector('#embedding-provider').value,document_mode:document.querySelector('#embedding-document-mode').value,base_url:document.querySelector('#embedding-url').value,model:document.querySelector('#embedding-model').value,api_key:document.querySelector('#embedding-key').value})});status.textContent=r.ok?'Configuración guardada.':'No se pudo guardar la configuración.';document.querySelector('#embedding-key').value='';loadSettings()};
async function download(url,name){const r=await api(url);if(!r.ok)return;const link=document.createElement('a');link.href=URL.createObjectURL(await r.blob());link.download=name;link.click();URL.revokeObjectURL(link.href)}
const metadataFields=[['invoice_number','Número de factura'],['invoice_date','Fecha'],['customer_code','Código de cliente'],['customer_tax_id','NIF de cliente'],['delivery_note_ids','Albaranes'],['concepts','Conceptos'],['currency','Moneda'],['total_payable','Total a pagar'],['invoice_total','Total factura'],['tax_base','Base imponible'],['vat','IVA']];
const metadataText=value=>{if(value==null||value===''||(Array.isArray(value)&&!value.length))return 'No detectado';return Array.isArray(value)?value.map(String).join(', '):typeof value==='object'?JSON.stringify(value):String(value)};
const metadataStyles=document.createElement('style');metadataStyles.textContent='.metadata-card{margin:12px 0;padding:16px;border:1px solid var(--line);border-radius:10px}.metadata-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:8px}.metadata-field{min-width:0;overflow-wrap:anywhere;padding:10px;border:1px solid var(--line);border-radius:8px}.metadata-field dt{color:var(--muted);font-size:.85rem}.metadata-field dd{margin:3px 0 0}.metadata-field small{display:block;color:var(--muted)}.metadata-card>p{overflow-wrap:anywhere}@media(min-width:640px){.metadata-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}';document.head.append(metadataStyles);
function renderDocumentMetadata(data,target){target.replaceChildren(el('p',`${data.filename} · ${data.invoice_id}`));if(!data.index_record_available)target.append(el('p','No hay un registro de metadatos en la generación visible.'));else if(!data.metadata_available)target.append(el('p','Documento indexado, pero sin metadatos genéricos disponibles.'));const grid=document.createElement('dl');grid.className='metadata-grid';for(const [name,label] of metadataFields){const field=document.createElement('div');field.className='metadata-field';field.append(el('dt',label));const value=data.metadata?.[name],definition=el('dd',metadataText(value)),confidence=data.confidence?.[name],evidence=data.evidence?.[name];if(confidence!=null)definition.append(el('small',`Confianza heurística: ${confidence}`));field.append(definition);if(evidence!=null&&evidence!==''&&(!Array.isArray(evidence)||evidence.length)){const details=document.createElement('details');details.append(el('summary','Evidencia'),el('p',metadataText(evidence)));field.append(details)}else if(value!=null&&value!==''&&(!Array.isArray(value)||value.length))field.append(el('small','Sin evidencia textual asociada.'));grid.append(field)}target.append(grid,el('p','Extracción heurística: los valores no son validación contable.'),originalLink(data,'Abrir PDF original'))}
function addMetadataAction(doc){const button=el('button','Ver metadatos');button.type='button';button.setAttribute('aria-expanded','false');const panel=el('div','');panel.className='metadata-card hidden';panel.setAttribute('aria-live','polite');let loaded=false,loading=false;button.onclick=async()=>{const opening=panel.classList.contains('hidden');panel.classList.toggle('hidden',!opening);button.textContent=opening?'Ocultar metadatos':'Ver metadatos';button.setAttribute('aria-expanded',String(opening));if(!opening||loaded||loading)return;loading=true;panel.replaceChildren(el('p','Cargando metadatos…'));try{const response=await api(`/api/documents/${encodeURIComponent(doc.id)}/metadata`),data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudieron cargar los metadatos');renderDocumentMetadata(data,panel);loaded=true}catch(error){panel.replaceChildren(el('p',error.message))}finally{loading=false}};return [button,panel]}
async function refresh(){if(!hasToken())return;const response=await api(`/api/documents?limit=${pageSize}&offset=${offset}`);if(!response.ok)return;const d=await response.json();const list=document.querySelector('#docs');if(list.querySelector('.metadata-card:not(.hidden)'))return;total=d.total;list.replaceChildren();for(const x of d.documents){const li=el('li','');const a=el('a',x.filename);a.href=`/api/documents/${x.id}/file`;a.onclick=e=>{e.preventDefault();download(a.href,x.filename)};li.append(a,document.createTextNode(` · ${x.status}${x.error?` · ${x.error}`:''} `));if(x.status==='failed'){const b=el('button','Reintentar');b.onclick=()=>api(`/api/documents/${x.id}/retry`,{method:'POST'}).then(refresh);li.append(b)}if(x.status==='indexed')li.append(...addMetadataAction(x));list.append(li)}document.querySelector('#page').textContent=`${total} documentos · ${total?offset+1:0}-${Math.min(offset+pageSize,total)}`;document.querySelector('#previous').disabled=offset===0;document.querySelector('#next').disabled=offset+pageSize>=total}
document.querySelector('#previous').onclick=()=>{offset=Math.max(0,offset-pageSize);refresh()};document.querySelector('#next').onclick=()=>{offset+=pageSize;refresh()};document.querySelector('#upload').onclick=async()=>{const files=document.querySelector('#files').files;const status=document.querySelector('#upload-status');if(!files.length){status.textContent='Selecciona al menos un PDF.';return}const body=new FormData();for(const f of files)body.append('files',f);status.textContent=`Enviando ${files.length} archivo(s)…`;try{const r=await api('/api/documents',{method:'POST',body});const d=await r.json();if(!r.ok)throw Error(d.detail||'Error al subir');status.textContent=`${d.accepted} en cola, ${d.duplicates} duplicados, ${d.rejected} rechazados.`;refresh()}catch(e){status.textContent=`No se pudo subir: ${e.message}`}};
document.querySelector('#search').onsubmit=async e=>{e.preventDefault();const list=document.querySelector('#results');const panel=document.querySelector('#results-panel');panel.classList.remove('hidden');list.replaceChildren(el('li','Buscando…'));try{const r=await api('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.querySelector('#query').value})});const d=await r.json();list.replaceChildren();if(!r.ok)throw Error(d.detail||'Error en búsqueda');if(!d.results.length)list.append(el('li','No se encontraron facturas.'));for(const x of d.results){const li=el('li','');const a=el('a',x.filename);a.href=x.file_url;a.onclick=ev=>{ev.preventDefault();download(a.href,x.filename)};li.append(a,document.createTextNode(` · relevancia ${x.score.toFixed(4)}`));list.append(li)}}catch(error){list.replaceChildren(el('li',error.message))}};
 const searchForm=document.querySelector('#search'),results=document.querySelector('#results'),resultsPanel=document.querySelector('#results-panel');let searchMode='semantic',filterOffset=0,filterPayload={};
 const modeBar=document.createElement('div');modeBar.className='tabs';modeBar.setAttribute('aria-label','Tipo de búsqueda');const modeButtons={};
 for(const [mode,label] of [['semantic','Facturas'],['evidence','Evidencia'],['filters','Filtros']]){const button=el('button',label);button.type='button';button.className='tab'+(mode===searchMode?' active':'');button.setAttribute('aria-pressed',String(mode===searchMode));modeBar.append(button);modeButtons[mode]=button;button.onclick=()=>{searchMode=mode;for(const [key,item] of Object.entries(modeButtons)){item.classList.toggle('active',key===mode);item.setAttribute('aria-pressed',String(key===mode))}searchForm.classList.toggle('hidden',mode==='filters');filterForm.classList.toggle('hidden',mode!=='filters');document.querySelector('#search button').textContent=mode==='evidence'?'Buscar evidencia':'Buscar facturas';document.querySelector('#query').placeholder=mode==='evidence'?'Ej. importe y concepto que quieres localizar':'Ej. mantenimiento de impresoras';resultsPanel.classList.add('hidden')}}
 searchForm.before(modeBar);
 const filterForm=document.createElement('form');filterForm.className='panel hidden';filterForm.setAttribute('aria-label','Filtros de facturas');const filterGrid=document.createElement('div');filterGrid.className='grid';
 const filterFields=[['invoice_number','Número de factura','text'],['customer_code','Código de cliente','text'],['customer_tax_id','NIF de cliente','text'],['delivery_note_id','Albarán','text'],['concept','Concepto','text'],['currency','Moneda','text'],['invoice_date_from','Fecha desde','date'],['invoice_date_to','Fecha hasta','date'],['total_payable_min','Total a pagar desde','number'],['total_payable_max','Total a pagar hasta','number']];
 for(const [name,label,type] of filterFields){const field=el('label',label);field.className='field';const input=document.createElement('input');input.name=name;input.type=type;if(type==='number'){input.step='0.01';input.min='0'}field.append(input);filterGrid.append(field)}
 const filterSubmit=el('button','Aplicar filtros');filterSubmit.type='submit';filterSubmit.className='primary';filterForm.append(filterGrid,filterSubmit);searchForm.after(filterForm);
 const filterPages=document.createElement('div');filterPages.className='actions';filterPages.hidden=true;const previousFilter=el('button','Anterior'),nextFilter=el('button','Siguiente'),filterPageLabel=el('span','');previousFilter.type=nextFilter.type='button';filterPages.append(previousFilter,filterPageLabel,nextFilter);results.after(filterPages);
 const setResultNote=text=>{resultsPanel.querySelector('.hint').textContent=text};
 const originalLink=(item,label=item.filename)=>{const link=el('a',label);link.href=item.file_url;link.onclick=event=>{event.preventDefault();download(link.href,item.filename)};return link};
 const showResults=()=>resultsPanel.classList.remove('hidden');
 function renderInvoice(item){const li=el('li','');li.append(originalLink(item),document.createTextNode(` · relevancia ${(item.score||0).toFixed(4)}`));const values=[['Nº',item.invoice_number],['Fecha',item.invoice_date],['Cliente',item.customer_code],['Total a pagar',item.total_payable==null?null:`${item.total_payable} ${item.currency||''}`]];const summary=values.filter(([,value])=>value).map(([label,value])=>`${label}: ${value}`).join(' · ');if(summary)li.append(el('p',summary));if(Object.keys(item.evidence||{}).length){const details=document.createElement('details'),title=el('summary','Confianza y evidencia extraída');details.append(title);for(const [field,value] of Object.entries(item.evidence)){const confidence=item.confidence?.[field];details.append(el('p',`${field}${confidence==null?'':` · confianza heurística ${confidence}`}: ${value}`))}li.append(details)}return li}
 function renderEvidence(item){const li=el('li','');li.append(el('p',item.content),originalLink(item),document.createTextNode(` · relevancia ${(item.score||0).toFixed(4)} · caracteres ${item.start_offset}–${item.end_offset}`));return li}
 async function loadFilterPage(offset){const response=await api('/api/invoices/filter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...filterPayload,limit:50,offset})});const data=await response.json();if(!response.ok)throw Error(data.detail||'Error al filtrar facturas');filterOffset=data.offset;filterPages.hidden=false;previousFilter.disabled=filterOffset===0;nextFilter.disabled=filterOffset+data.limit>=data.total;filterPageLabel.textContent=`${data.total} facturas · ${data.total?filterOffset+1:0}-${Math.min(filterOffset+data.limit,data.total)}`;return data}
 searchForm.onsubmit=async event=>{event.preventDefault();showResults();results.replaceChildren(el('li',searchMode==='evidence'?'Buscando pasajes…':'Buscando facturas…'));filterPages.hidden=true;try{const evidence=searchMode==='evidence',response=await api(evidence?'/api/search/evidence':'/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.querySelector('#query').value,...(evidence?{limit:10}:{})})}),data=await response.json();if(!response.ok)throw Error(data.detail||'Error en búsqueda');const items=evidence?data.evidence:data.results;results.replaceChildren();if(!items.length)results.append(el('li',evidence?'No se encontró evidencia textual.':'No se encontraron facturas.'));for(const item of items)results.append(evidence?renderEvidence(item):renderInvoice(item));setResultNote(evidence?'Offsets sobre el texto extraído; no corresponden a páginas del PDF. El ranking es relativo.':'La confianza de campos es heurística; la relevancia es un ranking, no una probabilidad.')}catch(error){results.replaceChildren(el('li',error.message))}};
 filterForm.onsubmit=async event=>{event.preventDefault();filterPayload={};for(const [name,,type] of filterFields){const value=filterForm.elements[name].value.trim();if(value!=='')filterPayload[name]=type==='number'?Number(value):value}if(!Object.keys(filterPayload).length){showResults();results.replaceChildren(el('li','Introduce al menos un criterio de filtro.'));filterPages.hidden=true;return}showResults();results.replaceChildren(el('li','Filtrando facturas…'));setResultNote('Los campos proceden de extracción heurística; revisa su evidencia antes de usarlos.');try{const data=await loadFilterPage(0);results.replaceChildren();if(!data.results.length)results.append(el('li','No se encontraron facturas con esos criterios.'));for(const item of data.results)results.append(renderInvoice(item))}catch(error){results.replaceChildren(el('li',error.message));filterPages.hidden=true}};
 previousFilter.onclick=async()=>{try{const data=await loadFilterPage(Math.max(0,filterOffset-50));results.replaceChildren(...data.results.map(renderInvoice))}catch(error){results.replaceChildren(el('li',error.message))}};nextFilter.onclick=async()=>{try{const data=await loadFilterPage(filterOffset+50);results.replaceChildren(...data.results.map(renderInvoice))}catch(error){results.replaceChildren(el('li',error.message))}};
 const evaluationTab=el('button','Evaluación');evaluationTab.type='button';evaluationTab.className='tab';const navigationTabs=document.querySelector('main.wrap > .tabs');navigationTabs.style.flexWrap='wrap';navigationTabs.append(evaluationTab);
 const evaluationView=document.createElement('section');evaluationView.id='evaluation-view';evaluationView.className='hidden';const evaluationPanel=document.createElement('div');evaluationPanel.className='panel';
 evaluationPanel.append(el('h2','Evaluación de evidencia'),el('p','Crea consultas y anota citas esperadas revisando el texto fuente. Los pasajes candidatos son orientación; no se etiquetan automáticamente. Las etiquetas se guardan en el backend local.'));
 const queryField=el('label','Nueva consulta');queryField.className='field';const evaluationQuery=document.createElement('input');evaluationQuery.type='text';evaluationQuery.maxLength=1000;evaluationQuery.placeholder='Ej. ¿Qué importe se pagó por el mantenimiento?';queryField.append(evaluationQuery);
 const caseSelectField=el('label','Casos guardados');caseSelectField.className='field';const caseSelect=document.createElement('select');styledSelect(caseSelect);caseSelectField.append(caseSelect);
 const caseActions=document.createElement('div');caseActions.className='actions';const addCaseButton=el('button','Añadir consulta'),deleteCaseButton=el('button','Eliminar consulta'),saveCasesButton=el('button','Guardar etiquetas');addCaseButton.type=deleteCaseButton.type=saveCasesButton.type='button';addCaseButton.className=saveCasesButton.className='primary';caseActions.append(addCaseButton,deleteCaseButton,saveCasesButton);
 evaluationPanel.append(queryField,caseActions,caseSelectField);
 const evidenceLabelPanel=document.createElement('div');evidenceLabelPanel.className='panel';evidenceLabelPanel.append(el('h3','Etiquetar evidencia esperada'),el('p','Selecciona una factura, revisa el texto extraído y añade una cita breve que responda a la consulta. También puedes comprobar el original.'));
 const invoiceSelectField=el('label','Factura indexada');invoiceSelectField.className='field';const invoiceSelect=document.createElement('select');styledSelect(invoiceSelect);invoiceSelectField.append(invoiceSelect);
 const invoicePager=document.createElement('div');invoicePager.className='actions';const previousInvoices=el('button','Facturas anteriores'),nextInvoices=el('button','Facturas siguientes'),invoicePageLabel=el('span','');previousInvoices.type=nextInvoices.type='button';invoicePager.append(previousInvoices,invoicePageLabel,nextInvoices);
 const extractedText=document.createElement('pre');extractedText.style.cssText='max-height:340px;overflow:auto;white-space:pre-wrap;border:1px solid var(--line);border-radius:8px;padding:14px;background:#0c1219';extractedText.textContent='Selecciona una factura para inspeccionar su texto extraído.';
 const copySelectionButton=el('button','Usar selección del texto'),quoteInput=document.createElement('textarea');quoteInput.rows=3;quoteInput.maxLength=2000;quoteInput.placeholder='Pega o selecciona una cita breve del texto fuente';quoteInput.style.cssText='width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:#0c1219;color:inherit;font:inherit';const addQuoteButton=el('button','Añadir cita a este caso');copySelectionButton.type=addQuoteButton.type='button';addQuoteButton.className='primary';const quoteActions=document.createElement('div');quoteActions.className='actions';quoteActions.append(copySelectionButton,addQuoteButton);
 const sourceActions=document.createElement('div');sourceActions.className='actions';evidenceLabelPanel.append(invoiceSelectField,invoicePager,extractedText,sourceActions,quoteInput,quoteActions);
 const labelsList=document.createElement('ul');evidenceLabelPanel.append(el('h3','Citas etiquetadas'),labelsList);
 const evaluationStatus=el('p','');evaluationStatus.setAttribute('role','status');const runEvaluationButton=el('button','Ejecutar evaluación');runEvaluationButton.type='button';runEvaluationButton.className='primary';const evaluationResults=document.createElement('div');evaluationView.append(evaluationPanel,evidenceLabelPanel,runEvaluationButton,evaluationStatus,evaluationResults);document.querySelector('main.wrap').append(evaluationView);
 let evaluationCases=[],selectedEvaluationCase=null,evaluationLoaded=false,invoiceOffset=0,invoiceTotal=0,indexedInvoices=[],extractedInvoiceContent='';
 const currentEvaluationCase=()=>evaluationCases.find(item=>item.id===selectedEvaluationCase);
 function renderEvaluationCases(){caseSelect.replaceChildren();for(const item of evaluationCases){const option=el('option',item.query);option.value=item.id;caseSelect.append(option)}caseSelect.value=selectedEvaluationCase||'';deleteCaseButton.disabled=!currentEvaluationCase();labelsList.replaceChildren();const current=currentEvaluationCase();for(const [index,label] of (current?.relevant_evidence||[]).entries()){const row=el('li',`${indexedInvoices.find(doc=>doc.id===label.invoice_id)?.filename||label.invoice_id}: “${label.quote}”`);const remove=el('button','Quitar');remove.type='button';remove.onclick=()=>{current.relevant_evidence.splice(index,1);renderEvaluationCases()};row.append(document.createTextNode(' '),remove);labelsList.append(row)}}
 async function loadEvaluationDocuments(offset=invoiceOffset){const response=await api(`/api/documents?status=indexed&limit=200&offset=${offset}`);const data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudo cargar la biblioteca');invoiceOffset=data.offset;invoiceTotal=data.total;indexedInvoices=data.documents;invoiceSelect.replaceChildren();for(const doc of indexedInvoices){const option=el('option',`${doc.filename} · ${doc.id.slice(0,8)}`);option.value=doc.id;invoiceSelect.append(option)}previousInvoices.disabled=invoiceOffset===0;nextInvoices.disabled=invoiceOffset+data.limit>=invoiceTotal;invoicePageLabel.textContent=`${invoiceTotal} facturas · ${invoiceTotal?invoiceOffset+1:0}-${Math.min(invoiceOffset+data.limit,invoiceTotal)}`;renderEvaluationCases();if(indexedInvoices.length)await loadExtractedText();else extractedText.textContent='No hay facturas indexadas.'}
 async function loadExtractedText(){sourceActions.replaceChildren();const id=invoiceSelect.value;extractedInvoiceContent='';if(!id){extractedText.textContent='Selecciona una factura indexada.';return}extractedText.textContent='Cargando texto extraído…';try{const response=await api(`/api/documents/${encodeURIComponent(id)}/text`);const data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudo cargar el texto');extractedInvoiceContent=data.content||'';extractedText.textContent=extractedInvoiceContent||'Este documento no tiene texto extraído (por ejemplo, puede ser una imagen sin OCR).';sourceActions.append(originalLink(data,'Abrir original'));}catch(error){extractedText.textContent=error.message}}
 async function loadEvaluationCases(){const response=await api('/api/evaluation/evidence/cases');const data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudieron cargar las etiquetas');evaluationCases=data.cases;selectedEvaluationCase=evaluationCases[0]?.id||null;renderEvaluationCases()}
 async function saveEvaluationCases(){const response=await api('/api/evaluation/evidence/cases',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({cases:evaluationCases})});const data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudieron guardar las etiquetas');evaluationStatus.textContent=`Guardados ${data.saved} casos en el backend local.`}
 async function openEvaluation(){evaluationTab.classList.add('active');for(const item of document.querySelectorAll('main.wrap > .tabs .tab'))if(item!==evaluationTab)item.classList.remove('active');for(const id of ['search-view','documents-view','settings-view'])document.querySelector(`#${id}`).classList.add('hidden');evaluationView.classList.remove('hidden');if(evaluationLoaded)return;evaluationStatus.textContent='Cargando casos y facturas…';try{await Promise.all([loadEvaluationCases(),loadEvaluationDocuments()]);evaluationLoaded=true;evaluationStatus.textContent='Etiquetas cargadas desde el backend local.'}catch(error){evaluationStatus.textContent=error.message}}
 evaluationTab.onclick=openEvaluation;caseSelect.onchange=()=>{selectedEvaluationCase=caseSelect.value;renderEvaluationCases()};invoiceSelect.onchange=()=>loadExtractedText();previousInvoices.onclick=()=>loadEvaluationDocuments(Math.max(0,invoiceOffset-200));nextInvoices.onclick=()=>loadEvaluationDocuments(invoiceOffset+200);
 addCaseButton.onclick=async()=>{const query=evaluationQuery.value.trim();if(!query){evaluationStatus.textContent='Escribe una consulta antes de añadir el caso.';return}const item={id:crypto.randomUUID(),query,relevant_evidence:[]};evaluationCases.push(item);selectedEvaluationCase=item.id;evaluationQuery.value='';renderEvaluationCases();try{await saveEvaluationCases()}catch(error){evaluationStatus.textContent=error.message}};
 deleteCaseButton.onclick=async()=>{evaluationCases=evaluationCases.filter(item=>item.id!==selectedEvaluationCase);selectedEvaluationCase=evaluationCases[0]?.id||null;renderEvaluationCases();try{await saveEvaluationCases()}catch(error){evaluationStatus.textContent=error.message}};saveCasesButton.onclick=async()=>{try{await saveEvaluationCases()}catch(error){evaluationStatus.textContent=error.message}};
 copySelectionButton.onclick=()=>{const selection=window.getSelection()?.toString().trim();if(selection)quoteInput.value=selection};addQuoteButton.onclick=async()=>{const current=currentEvaluationCase(),invoiceId=invoiceSelect.value,quote=quoteInput.value.trim(),normalize=text=>text.toLocaleLowerCase().split(/\\s+/).join(' ');if(!current){evaluationStatus.textContent='Añade o selecciona una consulta primero.';return}if(!invoiceId||!quote){evaluationStatus.textContent='Selecciona una factura y proporciona una cita.';return}if(!extractedInvoiceContent||!normalize(extractedInvoiceContent).includes(normalize(quote))){evaluationStatus.textContent='La cita no aparece en el texto extraído de esta factura.';return}if(current.relevant_evidence.some(item=>item.invoice_id===invoiceId&&item.quote===quote)){evaluationStatus.textContent='Esa cita ya está etiquetada en este caso.';return}current.relevant_evidence.push({invoice_id:invoiceId,quote});quoteInput.value='';renderEvaluationCases();try{await saveEvaluationCases()}catch(error){evaluationStatus.textContent=error.message}};
 const detailsList=(title,items,render)=>{const details=document.createElement('details');details.append(el('summary',title));const list=document.createElement('ul');for(const item of items)list.append(render(item));details.append(list);return details};
 const invoiceCandidateRow=item=>{const row=el('li','');row.append(originalLink(item),document.createTextNode(` · ${item.score.toFixed(4)}`));return row};
 const passageCandidateRow=item=>{const row=el('li','');row.append(el('p',item.content),originalLink(item),document.createTextNode(` · caracteres ${item.start_offset}–${item.end_offset} · score ${item.score.toFixed(4)}`));return row};
 runEvaluationButton.onclick=async()=>{evaluationStatus.textContent='Ejecutando búsquedas locales…';evaluationResults.replaceChildren();try{await saveEvaluationCases();const response=await api('/api/evaluation/evidence/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cases:evaluationCases})}),data=await response.json();if(!response.ok)throw Error(data.detail||'No se pudo ejecutar la evaluación');const s=data.summary;evaluationStatus.textContent=`${s.cases} casos · media Invoice@5/10 ${s.mean_invoice_recall_at_5.toFixed(3)}/${s.mean_invoice_recall_at_10.toFixed(3)} · Evidencia global@5/10 ${s.mean_global_evidence_recall_at_5.toFixed(3)}/${s.mean_global_evidence_recall_at_10.toFixed(3)} · Evidencia filtrada@5/10 ${s.mean_scoped_evidence_recall_at_5.toFixed(3)}/${s.mean_scoped_evidence_recall_at_10.toFixed(3)}`;for(const item of data.results){const panel=document.createElement('div');panel.className='panel';panel.append(el('h3',item.query),el('p',`Factura @5/10 ${item.invoice_recall_at_5.toFixed(3)}/${item.invoice_recall_at_10.toFixed(3)} · Evidencia global @5/10 ${item.global_evidence_recall_at_5.toFixed(3)}/${item.global_evidence_recall_at_10.toFixed(3)} · Evidencia filtrada @5/10 ${item.scoped_evidence_recall_at_5.toFixed(3)}/${item.scoped_evidence_recall_at_10.toFixed(3)} · ${item.latency_ms} ms`),el('small','La evidencia filtrada usa el invoice_id esperado: mide recuperación condicional, no resolución de factura.'));panel.append(detailsList('Facturas candidatas globales',item.invoice_top10,invoiceCandidateRow),detailsList('Pasajes globales',item.global_evidence_top10,passageCandidateRow));for(const group of item.scoped_evidence_top10)panel.append(detailsList(`Pasajes filtrados · factura ${group.invoice_id}`,group.evidence,passageCandidateRow));evaluationResults.append(panel)}}catch(error){evaluationStatus.textContent=error.message}};
 authUI(false);if(hasToken())document.querySelector('#save-token').click();setInterval(refresh,5000);
</script></html>'''
