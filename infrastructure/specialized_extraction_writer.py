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
MAX_SPECIALIZED_RESULT_TRANSFER_BYTES = 1_000_000


class SpecializedResultPayloadTooLarge(ValueError):
    pass


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


def fingerprint_extractor_set(components: tuple[Component, ...]) -> str:
    manifest = sorted((component.id, component.version) for component in components)
    return hashlib.sha256(_encode(manifest).encode()).hexdigest()


def _validate_evidence(evidence):
    if not isinstance(evidence, Evidence) or evidence.page_number < 1 or not evidence.quote:
        raise ValueError("evidence requires a positive page number and quote")
    for box in (evidence.bounds, evidence.label_bounds):
        if box and not all(isinstance(v, (int, float)) for v in (box.x0, box.top, box.x1, box.bottom)):
            raise ValueError("invalid evidence geometry")


def adapt_template_result(classification, extractor_id, extractor_version, fields, lines=(), *,
                          document_id, document_sha256, attempt_id, extractor_set_fingerprint=None,
                          idempotency_key=None, result_schema_version=1,
                          execution_mode="specialized", line_extractor=None, warnings=(),
                          operation_id="default", status="succeeded", error=None,
                          diagnostics_code=None):
    components = []
    if extractor_id and extractor_version:
        components.append(Component(extractor_id, extractor_version))
    if line_extractor:
        components.append(Component(line_extractor.id, line_extractor.version))
    if not components:
        components.append(Component(classification.detector_id, classification.detector_version))
    if extractor_set_fingerprint is None:
        extractor_set_fingerprint = fingerprint_extractor_set(tuple(components))
    if idempotency_key is None:
        idempotency_key = make_idempotency_key(
            document_id=document_id, document_sha256=document_sha256,
            execution_mode=execution_mode, detector_id=classification.detector_id,
            detector_version=classification.detector_version, template=classification.template,
            components=tuple(components), extractor_set_fingerprint=extractor_set_fingerprint,
            result_schema_version=result_schema_version, operation_id=operation_id,
        )
    reason = classification.reason if classification.template == "unknown" else None
    result = SpecializedResult(
        document_id, document_sha256, attempt_id, execution_mode, classification.detector_id,
        classification.detector_version, classification.template, tuple(components),
        extractor_set_fingerprint, result_schema_version, idempotency_key, tuple(fields), tuple(lines),
        status, tuple(warnings), error=error, abstention_reason=reason,
        diagnostics_code=diagnostics_code,
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


def make_idempotency_key(*, document_id, document_sha256, execution_mode, detector_id,
                         detector_version, template, components, extractor_set_fingerprint,
                         result_schema_version, operation_id="default"):
    """Build the same logical operation key after a caller process restarts."""
    material = {
        "key_version": 1, "document_id": document_id, "document_sha256": document_sha256,
        "execution_mode": execution_mode, "detector": (detector_id, detector_version),
        "template": template, "components": [(c.id, c.version) for c in components],
        "extractor_set_fingerprint": extractor_set_fingerprint,
        "result_schema_version": result_schema_version, "operation_id": operation_id,
    }
    digest = hashlib.sha256(_encode(material).encode()).hexdigest()
    return f"specialized-v1:{digest}"


def _pack_value(value):
    if value is None:
        return ["n"]
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, str):
        return ["s", value]
    if isinstance(value, int):
        return ["i", value]
    if isinstance(value, Decimal):
        return ["d", str(value)]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, tuple):
        return ["tuple", [_pack_value(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_pack_value(item) for item in value]]
    if isinstance(value, dict):
        return ["map", [[key, _pack_value(value[key])] for key in sorted(value)]]
    raise ValueError(f"unsupported transfer value: {type(value).__name__}")


def _unpack_value(value):
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ValueError("invalid specialized result transfer value")
    kind = value[0]
    if kind == "n" and len(value) == 1:
        return None
    if kind in {"b", "s", "i"} and len(value) == 2:
        expected = {"b": bool, "s": str, "i": int}[kind]
        if not isinstance(value[1], expected) or (kind == "i" and isinstance(value[1], bool)):
            raise ValueError("invalid specialized result transfer scalar")
        return value[1]
    if kind == "d" and len(value) == 2:
        return Decimal(value[1])
    if kind == "date" and len(value) == 2:
        return date.fromisoformat(value[1])
    if kind in {"tuple", "list"} and len(value) == 2 and isinstance(value[1], list):
        items = [_unpack_value(item) for item in value[1]]
        return tuple(items) if kind == "tuple" else items
    if kind == "map" and len(value) == 2 and isinstance(value[1], list):
        pairs = value[1]
        if any(not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
               for pair in pairs):
            raise ValueError("invalid specialized result transfer mapping")
        return {key: _unpack_value(item) for key, item in pairs}
    raise ValueError("invalid specialized result transfer tag")


def _pack_box(bounds):
    return None if bounds is None else [bounds.x0, bounds.top, bounds.x1, bounds.bottom]


def _unpack_box(bounds):
    if bounds is None:
        return None
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("invalid evidence bounds transfer")
    from domain.structured_extraction import BoundingBox
    return BoundingBox(*bounds)


def _pack_evidence(evidence):
    return {"page": evidence.page_number, "quote": evidence.quote, "bounds": _pack_box(evidence.bounds),
            "block": evidence.block_id, "label_quote": evidence.label_quote,
            "label_bounds": _pack_box(evidence.label_bounds), "method": evidence.method,
            "extractor_id": evidence.extractor_id, "extractor_version": evidence.extractor_version,
            "tokens": list(evidence.source_token_ids)}


def _unpack_evidence(value):
    return Evidence(value["page"], value["quote"], _unpack_box(value["bounds"]), value["block"],
                    value["label_quote"], _unpack_box(value["label_bounds"]), value["method"],
                    value["extractor_id"], value["extractor_version"], tuple(value["tokens"]))


def serialize_specialized_result(result: SpecializedResult, *, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES):
    result.validate()
    body = {
        "document_id": result.document_id, "document_sha256": result.document_sha256,
        "attempt_id": result.attempt_id, "execution_mode": result.execution_mode,
        "detector_id": result.detector_id, "detector_version": result.detector_version,
        "template": result.template, "components": [[c.id, c.version] for c in result.components],
        "extractor_set_fingerprint": result.extractor_set_fingerprint,
        "result_schema_version": result.result_schema_version, "idempotency_key": result.idempotency_key,
        "fields": [[item.name, _pack_value(item.value), [_pack_evidence(ev) for ev in item.evidence],
                    list(item.warnings), item.status, _pack_value(item.candidates)] for item in result.fields],
        "lines": [[item.description, _pack_value(item.values), [_pack_evidence(ev) for ev in item.evidence],
                   list(item.warnings)] for item in result.lines],
        "status": result.status, "warnings": list(result.warnings), "error": result.error,
        "abstention_reason": result.abstention_reason, "diagnostics_code": result.diagnostics_code,
    }
    try:
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("specialized result cannot be serialized safely") from exc
    if len(payload) > max_bytes:
        raise SpecializedResultPayloadTooLarge("specialized result exceeds the transfer size limit")
    return payload


def deserialize_specialized_result(payload: bytes, *, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES):
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise SpecializedResultPayloadTooLarge("specialized result transfer exceeds the size limit")
    body = json.loads(payload)
    required = {"document_id", "document_sha256", "attempt_id", "execution_mode", "detector_id",
                "detector_version", "template", "components", "extractor_set_fingerprint",
                "result_schema_version", "idempotency_key", "fields", "lines", "status", "warnings",
                "error", "abstention_reason", "diagnostics_code"}
    if not isinstance(body, dict) or set(body) != required:
        raise ValueError("invalid specialized result transfer shape")
    fields = tuple(ExtractedField(name, _unpack_value(value), tuple(_unpack_evidence(ev) for ev in evidence),
                                  tuple(warnings), status, _unpack_value(candidates))
                   for name, value, evidence, warnings, status, candidates in body["fields"])
    lines = tuple(ProposedLine(description, _unpack_value(values),
                               tuple(_unpack_evidence(ev) for ev in evidence), tuple(warnings))
                  for description, values, evidence, warnings in body["lines"])
    result = SpecializedResult(
        body["document_id"], body["document_sha256"], body["attempt_id"], body["execution_mode"],
        body["detector_id"], body["detector_version"], body["template"],
        tuple(Component(*component) for component in body["components"]),
        body["extractor_set_fingerprint"], body["result_schema_version"], body["idempotency_key"],
        fields, lines, body["status"], tuple(body["warnings"]), body["error"],
        body["abstention_reason"], body["diagnostics_code"],
    )
    return result.validate()


def serialize_shadow_extraction_result(result, *, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES):
    classification = result["classification"]
    line_extractor = result["line_extractor"]
    body = {
        "classification": {
            "template": classification.template, "detector_id": classification.detector_id,
            "detector_version": classification.detector_version,
            "signals": _pack_value(classification.signals), "reason": classification.reason,
        },
        "extractor_id": result["extractor_id"], "extractor_version": result["extractor_version"],
        "line_component": ([line_extractor.id, line_extractor.version] if line_extractor else None),
        "fields": [[item.name, _pack_value(item.value), [_pack_evidence(ev) for ev in item.evidence],
                    list(item.warnings), item.status, _pack_value(item.candidates)] for item in result["fields"]],
        "lines": [[item.description, _pack_value(item.values), [_pack_evidence(ev) for ev in item.evidence],
                   list(item.warnings)] for item in result["lines"]],
        "warnings": list(result["warnings"]), "status": result["status"],
        "error": result.get("error"), "diagnostics_code": result.get("diagnostics_code"),
    }
    try:
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("shadow extraction result cannot be serialized safely") from exc
    if len(payload) > max_bytes:
        raise SpecializedResultPayloadTooLarge("shadow extraction result exceeds the transfer size limit")
    return payload


def deserialize_shadow_extraction_result(payload: bytes, *, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES):
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise SpecializedResultPayloadTooLarge("shadow extraction transfer exceeds the size limit")
    body = json.loads(payload)
    required = {"classification", "extractor_id", "extractor_version", "line_component", "fields", "lines",
                "warnings", "status", "error", "diagnostics_code"}
    if not isinstance(body, dict) or set(body) != required:
        raise ValueError("invalid shadow extraction transfer shape")
    classification = body["classification"]
    if not isinstance(classification, dict) or set(classification) != {
            "template", "detector_id", "detector_version", "signals", "reason"}:
        raise ValueError("invalid shadow classification transfer shape")
    line_component = body["line_component"]
    if line_component is not None:
        line_component = Component(*line_component)
    fields = tuple(ExtractedField(name, _unpack_value(value), tuple(_unpack_evidence(ev) for ev in evidence),
                                  tuple(warnings), status, _unpack_value(candidates))
                   for name, value, evidence, warnings, status, candidates in body["fields"])
    lines = tuple(ProposedLine(description, _unpack_value(values),
                               tuple(_unpack_evidence(ev) for ev in evidence), tuple(warnings))
                  for description, values, evidence, warnings in body["lines"])
    return {
        "classification": {**classification, "signals": _unpack_value(classification["signals"])},
        "extractor_id": body["extractor_id"], "extractor_version": body["extractor_version"],
        "line_extractor": line_component, "fields": fields, "lines": lines,
        "warnings": tuple(body["warnings"]), "status": body["status"],
        "error": body["error"], "diagnostics_code": body["diagnostics_code"],
    }


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
