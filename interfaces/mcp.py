import base64
import binascii
import hashlib
import uuid
from contextlib import closing
from datetime import date

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Invoice indexing and semantic search", stateless_http=True, json_response=True)


def register_tools(*, max_bytes, connect, normalize_pdf, put_object, delete_object,
                   run_search, run_invoice_filter, run_invoice_evidence_search):
    @mcp.tool()
    def index_invoice(filename: str, content_base64: str) -> dict:
        """Encola un PDF individual enviado como base64 y devuelve su estado."""
        try:
            if len(content_base64) > 3_600_000:
                return {"status": "rejected", "error": "Para PDFs grandes, use la API REST multipart"}
            data = base64.b64decode(content_base64, validate=True)
            if len(data) > max_bytes():
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
                db.execute("INSERT INTO jobs(id,sha256,filename,object_key,status) VALUES(?,?,?,?,?)",
                           (doc_id, digest, filename, object_key, "queued"))
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
            row = db.execute("SELECT id,filename,status,error,created_at FROM jobs WHERE id=?",
                             (document_id,)).fetchone()
        return dict(row) if row else {"error": "Documento no encontrado"}

    @mcp.tool()
    def reindex_invoices(document_id: str | None = None) -> dict:
        """Reprocesa una factura por ID o todas las ya indexadas para extraer los campos estructurados."""
        with closing(connect()) as db:
            if document_id:
                result = db.execute(
                    "UPDATE jobs SET status='queued',error=NULL,active_generation_id=COALESCE(active_generation_id,'legacy') "
                    "WHERE id=? AND status IN ('indexed','failed')", (document_id,))
            else:
                result = db.execute("UPDATE jobs SET status='queued',error=NULL,"
                                    "active_generation_id=COALESCE(active_generation_id,'legacy') WHERE status='indexed'")
            db.commit()
            if result.rowcount:
                return {"requeued": result.rowcount,
                        **({"id": document_id, "status": "queued"} if document_id else {})}
            if document_id:
                row = db.execute("SELECT status FROM jobs WHERE id=?", (document_id,)).fetchone()
                return {"id": document_id, "status": row["status"]} if row else {"error": "Documento no encontrado"}
        return {"requeued": 0}

    @mcp.tool()
    def semantic_search(query: str, limit: int = 10) -> dict:
        """Busca facturas por significado y texto; limit debe estar entre 1 y 50."""
        if not 1 <= limit <= 50:
            return {"error": "limit debe estar entre 1 y 50"}
        try:
            return run_search(query, limit)
        except Exception as error:
            if isinstance(error, HTTPException):
                return {"error": error.detail}
            raise

    @mcp.tool()
    def filter_invoices(customer_code: str | None = None, customer_tax_id: str | None = None,
                        invoice_number: str | None = None, delivery_note_id: str | None = None,
                        concept: str | None = None, currency: str | None = None,
                        invoice_date_from: str | None = None, invoice_date_to: str | None = None,
                        total_payable_min: float | None = None, total_payable_max: float | None = None,
                        limit: int = 50, offset: int = 0) -> dict:
        """Filtra facturas por campos extraídos. Valores extraídos incluyen confianza y evidencia; no son verificación contable."""
        if not 1 <= limit <= 200 or offset < 0:
            return {"error": "limit debe estar entre 1 y 200 y offset no puede ser negativo"}
        try:
            start = date.fromisoformat(invoice_date_from) if invoice_date_from else None
            end = date.fromisoformat(invoice_date_to) if invoice_date_to else None
        except ValueError:
            return {"error": "Las fechas deben usar el formato YYYY-MM-DD"}
        if start and end and start > end:
            return {"error": "invoice_date_from no puede ser posterior a invoice_date_to"}
        if total_payable_min is not None and total_payable_max is not None and total_payable_min > total_payable_max:
            return {"error": "total_payable_min no puede superar total_payable_max"}
        try:
            return run_invoice_filter(
                customer_code=customer_code, customer_tax_id=customer_tax_id,
                invoice_number=invoice_number, delivery_note_id=delivery_note_id,
                concept=concept, currency=currency, invoice_date_from=invoice_date_from,
                invoice_date_to=invoice_date_to, total_payable_min=total_payable_min,
                total_payable_max=total_payable_max, limit=limit, offset=offset,
            )
        except Exception as error:
            if isinstance(error, HTTPException):
                return {"error": error.detail}
            raise

    @mcp.tool()
    def search_invoice_evidence(query: str, limit: int = 5, invoice_id: str | None = None) -> dict:
        """Recupera pasajes como evidencia con score de ranking relativo y offsets sobre el texto extraído (no páginas)."""
        if not query.strip() or not 1 <= limit <= 20:
            return {"error": "query no puede estar vacío y limit debe estar entre 1 y 20"}
        try:
            return run_invoice_evidence_search(query, limit, invoice_id)
        except Exception as error:
            if isinstance(error, HTTPException):
                return {"error": error.detail}
            raise

    return {"index_invoice": index_invoice, "get_invoice_status": get_invoice_status,
            "reindex_invoices": reindex_invoices, "semantic_search": semantic_search,
            "filter_invoices": filter_invoices, "search_invoice_evidence": search_invoice_evidence}
