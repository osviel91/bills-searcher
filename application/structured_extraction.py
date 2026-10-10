import uuid

from domain.structured_extraction import ExtractionExecution
from infrastructure.extraction_store import save_execution
from infrastructure.jobstore import connect
from infrastructure.object_store import get as get_object
from infrastructure.pdf_extraction import extract_pdf

EXTRACTOR_VERSION = "pdf-structure-v1"


def extract_document(document_id: str, *, db_path=None) -> str:
    with connect(db_path) as db:
        document = db.execute("SELECT id,object_key FROM jobs WHERE id=?", (document_id,)).fetchone()
    if not document:
        raise LookupError(f"Documento no encontrado: {document_id}")

    execution_id = str(uuid.uuid4())
    try:
        if not document["object_key"]:
            raise ValueError("El documento no tiene un original almacenado")
        data = get_object(document["object_key"]).readall()
        pages = extract_pdf(data)
        execution = ExtractionExecution(
            id=execution_id,
            document_id=document_id,
            extractor_version=EXTRACTOR_VERSION,
            status="succeeded",
            pages=pages,
        )
    except Exception as error:
        execution = ExtractionExecution(
            id=execution_id,
            document_id=document_id,
            extractor_version=EXTRACTOR_VERSION,
            status="failed",
            error=type(error).__name__,
        )
    save_execution(execution, db_path)
    return execution_id
