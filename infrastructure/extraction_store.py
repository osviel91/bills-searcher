import json
from dataclasses import asdict, is_dataclass
import uuid

from domain.structured_extraction import (
    BoundingBox,
    ExtractionExecution,
)
from infrastructure.jobstore import connect


def _json(value):
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _bounds(bounds: BoundingBox | None):
    return _json(vars(bounds)) if bounds else None


def save_execution(execution: ExtractionExecution, path=None):
    with connect(path) as db:
        db.execute("""INSERT INTO extraction_runs
            (id,document_id,extractor_version,status,warnings_json,error)
            VALUES(?,?,?,?,?,?)""", (
                execution.id, execution.document_id, execution.extractor_version,
                execution.status, _json(execution.warnings), execution.error,
            ))
        for page in execution.pages:
            db.execute("INSERT INTO extraction_pages(run_id,page_number,structure_json) VALUES(?,?,?)",
                       (execution.id, page.number, _json(page)))
        for item in execution.fields:
            db.execute("""INSERT INTO extraction_fields(run_id,name,value_json,warnings_json)
                VALUES(?,?,?,?)""", (
                    execution.id, item.name, _json(item.value), _json(item.warnings),
                ))
            for evidence in item.evidence:
                db.execute("""INSERT INTO extraction_evidence
                    (id,run_id,field_name,page_number,block_id,quote,bounds_json)
                    VALUES(?,?,?,?,?,?,?)""", (
                        str(uuid.uuid4()),
                    execution.id, item.name, evidence.page_number, evidence.block_id,
                        evidence.quote, _bounds(evidence.bounds),
                    ))
        for number, item in enumerate(execution.lines, start=1):
            db.execute("""INSERT INTO extraction_lines(run_id,line_number,description,values_json,warnings_json)
                VALUES(?,?,?,?,?)""", (
                    execution.id, number, item.description, _json(item.values), _json(item.warnings),
                ))
            for evidence in item.evidence:
                db.execute("""INSERT INTO extraction_evidence
                    (id,run_id,line_number,page_number,block_id,quote,bounds_json)
                    VALUES(?,?,?,?,?,?,?)""", (
                        str(uuid.uuid4()),
                    execution.id, number, evidence.page_number, evidence.block_id,
                        evidence.quote, _bounds(evidence.bounds),
                    ))
