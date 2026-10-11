"""Isolated PostgreSQL writer for complete, already-computed template results."""

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from domain.structured_extraction import Evidence, ExtractedField, ProposedLine
from infrastructure.jobstore import connect

_STATUSES = {"confirmed", "absent", "ambiguous", "not_applicable", "unreadable",
             "conflict", "insufficient_evidence", "unverified_semantics"}
_TERMINAL = {"succeeded", "failed", "discarded_fenced"}
_SAFE_DIAGNOSTIC = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")
RESULT_FINGERPRINT_VERSION = 1


@dataclass(frozen=True)
class Component:
    id: str
    version: str


@dataclass(frozen=True)
class SpecializedResult:
    document_id: str
    document_sha256: str
    attempt_id: str
    execution_mode: str
    detector_id: str
    detector_version: str
    template: str
    components: tuple[Component, ...]
    extractor_set_fingerprint: str
    result_schema_version: int
    idempotency_key: str
    fields: tuple[ExtractedField, ...]
    lines: tuple[ProposedLine, ...]
    status: str
    warnings: tuple[str, ...] = ()
    error: str | None = None
    abstention_reason: str | None = None
    diagnostics_code: str | None = None
    run_id: str = ""

    def validate(self):
        for name in ("document_id", "attempt_id", "execution_mode", "detector_id",
                     "detector_version", "template", "extractor_set_fingerprint", "idempotency_key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not re.fullmatch(r"[0-9a-f]{64}", self.document_sha256):
            raise ValueError("document_sha256 must be lowercase SHA-256 hex")
        if self.status not in _TERMINAL:
            raise ValueError("specialized results must be terminal")
        if self.status != "succeeded" and (self.fields or self.lines):
            raise ValueError("failed or fenced runs cannot persist partial observations")
        if self.template == "unknown" and (self.status != "succeeded" or not self.abstention_reason):
            raise ValueError("unknown template requires succeeded status and abstention_reason")
        if self.template != "unknown" and self.status == "succeeded" and not self.fields and not self.lines:
            raise ValueError("successful recognized result must contain observations")
        if not isinstance(self.result_schema_version, int) or self.result_schema_version < 1:
            raise ValueError("result_schema_version must be a positive integer")
        if not self.components or any(not c.id.strip() or not c.version.strip() for c in self.components):
            raise ValueError("every extractor component must have an id and version")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("field names must be unique")
        for field in self.fields:
            if not field.name or field.status not in _STATUSES:
                raise ValueError("each field requires a name and explicit valid observation status")
            _json_value(field.value)
            for value in field.candidates:
                _json_value(value)
            for evidence in field.evidence:
                _validate_evidence(evidence)
        for line in self.lines:
            _json_value(line.values)
            for evidence in line.evidence:
                _validate_evidence(evidence)
        for message in (*self.warnings, *((self.error,) if self.error else ())):
            if not isinstance(message, str) or not _SAFE_DIAGNOSTIC.fullmatch(message):
                raise ValueError("diagnostics must be safe codes, not free text")
        for message in (self.abstention_reason, self.diagnostics_code):
            if message is not None and (not isinstance(message, str) or not _SAFE_DIAGNOSTIC.fullmatch(message)):
                raise ValueError("diagnostics must be safe codes, not free text")
        if not self.run_id:
            object.__setattr__(self, "run_id", str(uuid.uuid4()))
        return self


@dataclass(frozen=True)
class WriteResult:
    run_id: str
    attempt_id: str
    reused: bool


def _json_value(value):
    if value is None or isinstance(value, (str, int, bool, Decimal, date)):
        return
    if isinstance(value, float):
        raise ValueError("binary floats are not accepted in observations")
    if isinstance(value, (tuple, list)):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("observation object keys must be strings")
        for item in value.values():
            _json_value(item)
        return
    raise ValueError(f"unsupported observation value: {type(value).__name__}")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (Decimal, date)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _bounds(value):
    return _encode(value) if value else None


def _validate_evidence(evidence):
    if not isinstance(evidence, Evidence) or evidence.page_number < 1 or not evidence.quote:
        raise ValueError("evidence requires a positive page number and quote")
    for box in (evidence.bounds, evidence.label_bounds):
        if box and not all(isinstance(v, (int, float)) for v in (box.x0, box.top, box.x1, box.bottom)):
            raise ValueError("invalid evidence geometry")


def adapt_template_result(classification, extractor_id, extractor_version, fields, lines=(), *,
                          document_id, document_sha256, attempt_id, idempotency_key,
                          extractor_set_fingerprint, result_schema_version=1,
                          execution_mode="specialized", line_extractor=None, warnings=()):
    components = []
    if extractor_id and extractor_version:
        components.append(Component(extractor_id, extractor_version))
    if line_extractor:
        components.append(Component(line_extractor.id, line_extractor.version))
    if not components:
        components.append(Component(classification.detector_id, classification.detector_version))
    reason = classification.reason if classification.template == "unknown" else None
    result = SpecializedResult(
        document_id, document_sha256, attempt_id, execution_mode, classification.detector_id,
        classification.detector_version, classification.template, tuple(components),
        extractor_set_fingerprint, result_schema_version, idempotency_key, tuple(fields), tuple(lines),
        "succeeded", tuple(warnings), abstention_reason=reason,
    )
    return result.validate()


def _payload_fingerprint(result):
    material = {
        "fingerprint_version": RESULT_FINGERPRINT_VERSION,
        "template": result.template, "components": [(c.id, c.version) for c in result.components],
        "detector": (result.detector_id, result.detector_version),
        "schema": result.result_schema_version, "fields": result.fields, "lines": result.lines,
        "status": result.status, "warnings": result.warnings, "abstention": result.abstention_reason,
        "diagnostics": result.diagnostics_code, "error": result.error,
    }
    return hashlib.sha256(_encode(material).encode()).hexdigest()


def write_specialized_result(result: SpecializedResult, path=None) -> WriteResult:
    result.validate()
    payload_fingerprint = _payload_fingerprint(result)
    with connect(path) as db:
        if not db.postgres:
            raise RuntimeError("specialized extraction writer requires PostgreSQL")
        job = db.execute("SELECT id,sha256,status,attempt_id FROM jobs WHERE id=? FOR UPDATE",
                         (result.document_id,)).fetchone()
        if not job or job["sha256"] != result.document_sha256 or job["status"] != "indexed" or \
                job["attempt_id"] != result.attempt_id:
            raise LookupError("document publication or extraction attempt is no longer authoritative")
        existing = db.execute("SELECT id,attempt_id,document_sha256,extractor_set_fingerprint,execution_mode,"
                              "result_schema_version,result_fingerprint_version,result_fingerprint,status "
                              "FROM extraction_runs "
                              "WHERE document_id=? AND idempotency_key=?",
                              (result.document_id, result.idempotency_key)).fetchone()
        if existing:
            expected = (result.document_sha256, result.extractor_set_fingerprint, result.execution_mode,
                        result.result_schema_version, RESULT_FINGERPRINT_VERSION, payload_fingerprint, result.status)
            actual = (existing["document_sha256"], existing["extractor_set_fingerprint"],
                      existing["execution_mode"], existing["result_schema_version"],
                      existing["result_fingerprint_version"], existing["result_fingerprint"], existing["status"])
            if actual != expected:
                raise ValueError("idempotency key conflicts with a different result")
            return WriteResult(existing["id"], existing["attempt_id"], True)

        component_versions = _encode([{"id": c.id, "version": c.version} for c in result.components])
        db.execute("""INSERT INTO extraction_runs
            (id,document_id,extractor_version,status,warnings_json,error,document_sha256,attempt_id,
              detector_id,detector_version,template,extractor_id,extractor_set_fingerprint,execution_mode,
              result_schema_version,idempotency_key,abstention_reason,diagnostics_code,finished_at,
              result_fingerprint,result_fingerprint_version)
            VALUES(?,?,?,'running',?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
            (result.run_id, result.document_id, component_versions, _encode(result.warnings), result.error,
             result.document_sha256, result.attempt_id, result.detector_id, result.detector_version,
             result.template, ",".join(component.id for component in result.components),
             result.extractor_set_fingerprint, result.execution_mode, result.result_schema_version,
             result.idempotency_key, result.abstention_reason, result.diagnostics_code, payload_fingerprint,
             RESULT_FINGERPRINT_VERSION))

        evidence_items = [ev for field in result.fields for ev in field.evidence]
        evidence_items.extend(ev for line in result.lines for ev in line.evidence)
        for page_number in sorted({ev.page_number for ev in evidence_items}):
            db.execute("INSERT INTO extraction_pages(run_id,page_number,structure_json) VALUES(?,?,?)",
                       (result.run_id, page_number, "{}"))
        for field in result.fields:
            db.execute("""INSERT INTO extraction_fields
                (run_id,name,value_json,warnings_json,observation_status,candidates_json)
                VALUES(?,?,?,?,?,?)""", (result.run_id, field.name, _encode(field.value),
                  _encode(field.warnings), field.status, _encode(field.candidates)))
        for ordinal, line in enumerate(result.lines, 1):
            db.execute("INSERT INTO extraction_lines(run_id,line_number,description,values_json,warnings_json) "
                       "VALUES(?,?,?,?,?)", (result.run_id, ordinal, line.description,
                       _encode(line.values), _encode(line.warnings)))
        for field in result.fields:
            for ev in field.evidence:
                _insert_evidence(db, result.run_id, field.name, None, ev)
        for ordinal, line in enumerate(result.lines, 1):
            for ev in line.evidence:
                _insert_evidence(db, result.run_id, None, ordinal, ev)
        db.execute("UPDATE extraction_runs SET status=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
                   (result.status, result.run_id))
        return WriteResult(result.run_id, result.attempt_id, False)


def _insert_evidence(db, run_id, field_name, line_number, evidence):
    db.execute("""INSERT INTO extraction_evidence
        (id,run_id,field_name,line_number,page_number,block_id,quote,bounds_json,label_quote,
         label_bounds_json,method,extractor_id,extractor_version,source_token_ids_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), run_id, field_name, line_number, evidence.page_number, evidence.block_id,
         evidence.quote, _bounds(evidence.bounds), evidence.label_quote, _bounds(evidence.label_bounds),
         evidence.method, evidence.extractor_id, evidence.extractor_version, _encode(evidence.source_token_ids)))
