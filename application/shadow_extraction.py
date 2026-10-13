"""Best-effort AS/400 extraction diagnostics; never an operational result."""

import logging
import multiprocessing
import os
import time

from application.extractor_registry import GENERIC, template_extractor_for, template_line_extractor_for
from application.template_detection import TemplateDetection, detect_template
from infrastructure.pdf_extraction import extract_pdf_tokens

LOGGER = logging.getLogger(__name__)
TIMEOUT_SECONDS = 5
_IPC_ENVELOPE_RESERVE_BYTES = 16_384

# Only these pairs have equivalent meanings in both extractors. Financial totals
# are intentionally excluded because the specialized layouts distinguish roles.
FIELD_EQUIVALENTS = {
    "invoice_number": "invoice_number",
    "issue_date": "issue_date",
    "issuer_tax_id": "issuer_tax_id",
    "customer_tax_id": "receiver_tax_id",
    "currency": "currency",
    "tax_base": "tax_base",
    "vat": "tax_amount",
}


def enabled():
    return os.getenv("AS400_SHADOW_ENABLED", "false").strip().lower() == "true"


def persistence_enabled():
    return os.getenv("AS400_SHADOW_PERSISTENCE_ENABLED", "false").strip().lower() == "true"


def _canonical(value):
    if hasattr(value, "normalize"):
        return value.normalize()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip().upper()
    return value


def compare_fields(generic_fields, specialized_fields, generic_lines, specialized_lines):
    generic = {field.name: field.value for field in generic_fields}
    specialized = {field.name: field.value for field in specialized_fields}
    both, specialized_only, generic_only, normalization = [], [], [], []
    for generic_name, specialized_name in FIELD_EQUIVALENTS.items():
        generic_value, specialized_value = generic.get(generic_name), specialized.get(specialized_name)
        if generic_value is not None and specialized_value is not None:
            both.append(generic_name)
            if (str(generic_value) != str(specialized_value)
                    and _canonical(generic_value) == _canonical(specialized_value)):
                normalization.append(generic_name)
        elif specialized_value is not None:
            specialized_only.append(generic_name)
        elif generic_value is not None:
            generic_only.append(generic_name)
    return {
        "fields_in_both": tuple(both),
        "fields_specialized_only": tuple(specialized_only),
        "fields_generic_only": tuple(generic_only),
        "normalization_discrepancies": tuple(normalization),
        "generic_line_count": len(generic_lines),
        "specialized_line_count": len(specialized_lines),
        "line_count_difference": len(specialized_lines) - len(generic_lines),
    }


def _diagnose(pages, data, generic_fields, generic_lines, on_classification=None,
              include_persistence=False):
    try:
        classification = detect_template(pages)
    except Exception as exc:
        return {"status": "detector_error", "error_type": type(exc).__name__,
                "template": "unknown", "abstention": "detector-error",
                "detector_version": "as400-template-router@1",
                "extractor_versions": (f"{GENERIC.id}@{GENERIC.version}",),
                "fields_produced": 0, "lines_produced": 0, "warnings": 0}

    detector_version = f"{classification.detector_id}@{classification.detector_version}"
    result = {"detector_version": detector_version, "template": classification.template,
              "extractor_versions": (f"{GENERIC.id}@{GENERIC.version}",)}
    if classification.template in {"T01", "T02"}:
        result["extractor_versions"] += (
            f"as400-{classification.template.lower()}-header-summary@1",
            f"as400-{classification.template.lower()}-lines@1",
        )
    elif classification.reason:
        result["abstention"] = classification.reason
    result.update(fields_produced=0, lines_produced=0, warnings=0)
    if on_classification:
        on_classification(result)
    if classification.template not in {"T01", "T02"}:
        result.update(status="abstained", abstention=classification.reason or "unknown-template",
                      fields_produced=0, lines_produced=0, warnings=0)
        if include_persistence:
            result["_specialized_result"] = {
                "classification": classification, "extractor_id": None,
                "extractor_version": None, "line_extractor": None,
                "fields": (), "lines": (), "warnings": (), "status": "succeeded",
            }
        return result

    fields, lines, field_warnings, line_warnings = (), (), (), ()
    header_extractor = line_extractor = None
    try:
        header_extractor = template_extractor_for(classification.template)
        line_extractor = template_line_extractor_for(classification.template)
        token_pages = extract_pdf_tokens(data)
        fields, _, field_warnings = header_extractor.extract(pages, token_pages)
        _, lines, line_warnings = line_extractor.extract(pages, token_pages)
    except Exception as exc:
        result.update(status="extractor_error", error_type=type(exc).__name__,
                      fields_produced=len(fields), lines_produced=len(lines),
                      warnings=len(field_warnings) + len(line_warnings),
                      extractor_versions=result["extractor_versions"] + (
                          f"as400-{classification.template.lower()}-header-summary@1",
                          f"as400-{classification.template.lower()}-lines@1"))
        if include_persistence and header_extractor and line_extractor:
            result["_specialized_result"] = {
                "classification": classification, "extractor_id": header_extractor.id,
                "extractor_version": header_extractor.version, "line_extractor": line_extractor,
                "fields": (), "lines": (), "warnings": (), "status": "failed",
                "error": type(exc).__name__, "diagnostics_code": "shadow-extractor-error",
            }
        return result

    result.update(
        status="succeeded",
        extractor_versions=result["extractor_versions"] + (f"{header_extractor.id}@{header_extractor.version}",
                            f"{line_extractor.id}@{line_extractor.version}"),
        fields_produced=len(fields), lines_produced=len(lines),
        warnings=len(field_warnings) + len(line_warnings),
        comparison=compare_fields(generic_fields, fields, generic_lines, lines),
    )
    if include_persistence:
        result["_specialized_result"] = {
            "classification": classification, "extractor_id": header_extractor.id,
            "extractor_version": header_extractor.version, "line_extractor": line_extractor,
            "fields": fields, "lines": lines,
            "warnings": field_warnings + line_warnings, "status": "succeeded",
        }
    return result


def _diagnose_in_child(send, pages, data, generic_fields, generic_lines, include_persistence):
    try:
        extraction_started = time.monotonic()
        send.send({"status": "running", "detector_version": "as400-template-router@1",
                   "template": "unknown", "abstention": "classification-not-completed",
                   "extractor_versions": (f"{GENERIC.id}@{GENERIC.version}",),
                   "fields_produced": 0, "lines_produced": 0, "warnings": 0})
        result = _diagnose(pages, data, generic_fields, generic_lines,
                           on_classification=lambda classification: send.send(classification | {"status": "running"}),
                           include_persistence=include_persistence)
        extraction_duration_ms = round((time.monotonic() - extraction_started) * 1000)
        result["extraction_duration_ms"] = extraction_duration_ms
        payload = result.pop("_specialized_result", None)
        if include_persistence:
            if payload is None:
                result["persistence_status"] = "not_attempted"
            else:
                from infrastructure.specialized_extraction_writer import (
                    MAX_SPECIALIZED_RESULT_TRANSFER_BYTES, SpecializedResultPayloadTooLarge,
                    serialize_shadow_extraction_result,
                )
                try:
                    transfer_payload = serialize_shadow_extraction_result(
                        payload, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES - _IPC_ENVELOPE_RESERVE_BYTES)
                except SpecializedResultPayloadTooLarge as exc:
                    result.update(persistence_status="payload_too_large",
                                  persistence_error_type=type(exc).__name__,
                                  persistence_error_code="specialized-payload-limit-exceeded")
                except Exception as exc:
                    result.update(persistence_status="adapter_error",
                                  persistence_error_type=type(exc).__name__,
                                  persistence_error_code="specialized-serialization-error")
                else:
                    result.update(persistence_status="prepared", persistence_payload=transfer_payload)
        else:
            result["persistence_status"] = "disabled"
        send.send(result)
    except Exception as exc:
        send.send({"status": "worker_error", "error_type": type(exc).__name__})
    finally:
        send.close()


def _document_sha256(document_id, attempt_id, db_path):
    from contextlib import closing
    from infrastructure.jobstore import connect

    with closing(connect(db_path)) as db:
        row = db.execute("SELECT sha256 FROM jobs WHERE id=? AND status='indexed' AND attempt_id=?",
                         (document_id, attempt_id)).fetchone()
    return row["sha256"] if row else None


def _adapt_shadow_extraction_result(payload, context):
    from infrastructure.specialized_extraction_writer import adapt_template_result
    try:
        classification = TemplateDetection(**payload["classification"])
        specialized = adapt_template_result(
            classification, payload["extractor_id"], payload["extractor_version"],
            payload["fields"], payload["lines"], document_id=context["document_id"],
            document_sha256=context["document_sha256"], attempt_id=context["attempt_id"],
            execution_mode="shadow-specialized", line_extractor=payload["line_extractor"],
            warnings=payload["warnings"], status=payload["status"], error=payload.get("error"),
            diagnostics_code=payload.get("diagnostics_code"), operation_id="as400-shadow-v1",
        )
    except Exception as exc:
        return {"persistence_status": "adapter_error",
                "persistence_error_type": type(exc).__name__,
                "persistence_error_code": "specialized-adapter-error"}, None
    return None, specialized


def _persist_specialized_result(specialized, db_path=None):
    from infrastructure.specialized_extraction_writer import write_specialized_result
    import psycopg

    started = time.monotonic()
    diagnostic = {"persistence_status": "error", "idempotency_key": specialized.idempotency_key}
    try:
        written = write_specialized_result(specialized, db_path)
        diagnostic.update(persistence_status="reused" if written.reused else "created",
                          persistence_run_id=written.run_id,
                          persistence_attempt_id=written.attempt_id)
    except LookupError as exc:
        diagnostic.update(persistence_status="stale", persistence_error_type=type(exc).__name__,
                          persistence_error_code="attempt-not-authoritative")
    except ValueError as exc:
        diagnostic.update(persistence_status="conflict", persistence_error_type=type(exc).__name__,
                          persistence_error_code="idempotency-conflict")
    except psycopg.OperationalError as exc:
        diagnostic.update(persistence_status="indeterminate", persistence_error_type=type(exc).__name__,
                          persistence_error_code="commit-outcome-unknown")
    except psycopg.Error as exc:
        code = "transaction-rolled-back" if (exc.sqlstate or "").startswith(("22", "23", "40")) \
            else "specialized-persistence-error"
        diagnostic.update(persistence_status="error", persistence_error_type=type(exc).__name__,
                          persistence_error_code=code)
    except Exception as exc:
        diagnostic.update(persistence_status="error", persistence_error_type=type(exc).__name__,
                          persistence_error_code="specialized-persistence-error")
    diagnostic["persistence_duration_ms"] = round((time.monotonic() - started) * 1000)
    return diagnostic


def _run_isolated(target, args, timeout):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args), daemon=True)
    try:
        process.start()
        sender.close()
        last_result = None
        receiver_eof = False
        deadline = time.monotonic() + max(0, timeout)
        while process.is_alive() and time.monotonic() < deadline:
            if not receiver_eof and receiver.poll(min(0.05, max(0, deadline - time.monotonic()))):
                try:
                    last_result = receiver.recv()
                except EOFError:
                    receiver_eof = True
            else:
                process.join(0.01)
        if process.is_alive():
            process.terminate()
            process.join()
            while not receiver_eof and receiver.poll():
                try:
                    last_result = receiver.recv()
                except EOFError:
                    receiver_eof = True
                    break
            return {**(last_result or {}), "status": "timeout", "error_type": "TimeoutError"}
        while not receiver_eof and receiver.poll():
            try:
                last_result = receiver.recv()
            except EOFError:
                receiver_eof = True
                break
        return ({**last_result, "status": "worker_error", "error_type": "ChildProcessError"}
                if last_result is None else last_result)
    except Exception as exc:
        if process.is_alive():
            process.terminate()
            process.join()
        return {"status": "worker_error", "error_type": type(exc).__name__}
    finally:
        receiver.close()
        sender.close()


def _still_current(document_id, attempt_id, db_path):
    from contextlib import closing
    from infrastructure.jobstore import connect

    with closing(connect(db_path)) as db:
        row = db.execute("SELECT status,attempt_id FROM jobs WHERE id=?", (document_id,)).fetchone()
    return bool(row and row["status"] == "indexed" and row["attempt_id"] == attempt_id)


def run(document_id, attempt_id, pages, data, generic_fields, generic_lines, *, db_path=None,
        timeout=None):
    """Run in a killable child so a slow PDF/parser cannot hold the worker indefinitely."""
    timeout = TIMEOUT_SECONDS if timeout is None else timeout
    if not enabled() or not attempt_id or not _still_current(document_id, attempt_id, db_path):
        return None

    persist = persistence_enabled()
    persistence_context = None
    if persist:
        try:
            sha256 = _document_sha256(document_id, attempt_id, db_path)
            if sha256:
                persistence_context = {"document_id": document_id, "document_sha256": sha256,
                                       "attempt_id": attempt_id, "db_path": db_path}
        except Exception:
            LOGGER.warning("AS/400 shadow persistence identity lookup failed", extra={
                "as400_shadow": {"document_id": document_id, "attempt_id": attempt_id,
                                 "persistence_status": "not_attempted",
                                 "persistence_error_code": "document-identity-lookup-failed"}})

    started = time.monotonic()
    result = _run_isolated(_diagnose_in_child,
                           (pages, data, generic_fields, generic_lines, persist), timeout)
    shadow_duration_ms = round((time.monotonic() - started) * 1000)

    if not _still_current(document_id, attempt_id, db_path):
        return None
    transfer_payload = result.pop("persistence_payload", None)
    if persist and result.get("persistence_status") == "prepared" and \
            result.get("status") != "timeout" and transfer_payload is not None:
        from infrastructure.specialized_extraction_writer import (
            MAX_SPECIALIZED_RESULT_TRANSFER_BYTES, SpecializedResultPayloadTooLarge,
            deserialize_shadow_extraction_result,
        )
        try:
            extraction_result = deserialize_shadow_extraction_result(
                transfer_payload, max_bytes=MAX_SPECIALIZED_RESULT_TRANSFER_BYTES - _IPC_ENVELOPE_RESERVE_BYTES)
        except SpecializedResultPayloadTooLarge as exc:
            result.update(persistence_status="payload_too_large", persistence_error_type=type(exc).__name__,
                          persistence_error_code="specialized-payload-limit-exceeded")
        except Exception as exc:
            result.update(persistence_status="adapter_error", persistence_error_type=type(exc).__name__,
                          persistence_error_code="specialized-deserialization-error")
        else:
            if not persistence_context:
                result.update(persistence_status="not_attempted",
                              persistence_error_code="document-identity-unavailable")
            else:
                adapter_error, specialized = _adapt_shadow_extraction_result(extraction_result,
                                                                             persistence_context)
                if adapter_error:
                    result.update(adapter_error)
                else:
                    result.update(_persist_specialized_result(specialized, db_path))
    elif persist and result.get("persistence_status") == "prepared" and result.get("status") == "timeout":
        result.update(persistence_status="not_attempted",
                      persistence_error_code="shadow-extraction-timeout")
    elif persist and result.get("persistence_status") == "prepared":
        result.update(persistence_status="error", persistence_error_code="specialized-payload-missing")
    if persist and result.get("persistence_status") == "running":
        result.update(persistence_status="indeterminate", persistence_error_type="TimeoutError",
                      persistence_error_code="commit-outcome-unknown")
    result.setdefault("persistence_status", "not_attempted" if persist else "disabled")
    result.setdefault("detector_version", "as400-template-router@1")
    result.setdefault("template", "unknown")
    result.setdefault("abstention", "classification-not-completed")
    result.setdefault("extractor_versions", (f"{GENERIC.id}@{GENERIC.version}",))
    result.setdefault("fields_produced", 0)
    result.setdefault("lines_produced", 0)
    result.setdefault("warnings", 0)
    result.setdefault("extraction_duration_ms", shadow_duration_ms)
    result.setdefault("persistence_duration_ms", 0)
    result["shadow_status"] = result.get("status", "unknown")
    result["persistence_enabled"] = persist
    result.update(document_id=document_id, attempt_id=attempt_id,
                  shadow_duration_ms=shadow_duration_ms,
                  duration_ms=round((time.monotonic() - started) * 1000))
    LOGGER.info("AS/400 shadow extraction diagnostic", extra={"as400_shadow": result})
    return result
