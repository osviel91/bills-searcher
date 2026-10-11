"""Best-effort AS/400 extraction diagnostics; never an operational result."""

import logging
import multiprocessing
import os
import time

from application.extractor_registry import GENERIC, template_extractor_for, template_line_extractor_for
from application.template_detection import detect_template
from infrastructure.pdf_extraction import extract_pdf_tokens

LOGGER = logging.getLogger(__name__)
TIMEOUT_SECONDS = 5

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


def _diagnose(pages, data, generic_fields, generic_lines, on_classification=None):
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
        return result

    fields, lines, field_warnings, line_warnings = (), (), (), ()
    try:
        token_pages = extract_pdf_tokens(data)
        header_extractor = template_extractor_for(classification.template)
        line_extractor = template_line_extractor_for(classification.template)
        fields, _, field_warnings = header_extractor.extract(pages, token_pages)
        _, lines, line_warnings = line_extractor.extract(pages, token_pages)
    except Exception as exc:
        result.update(status="extractor_error", error_type=type(exc).__name__,
                      fields_produced=len(fields), lines_produced=len(lines),
                      warnings=len(field_warnings) + len(line_warnings),
                      extractor_versions=result["extractor_versions"] + (
                          f"as400-{classification.template.lower()}-header-summary@1",
                          f"as400-{classification.template.lower()}-lines@1"))
        return result

    result.update(
        status="succeeded",
        extractor_versions=result["extractor_versions"] + (f"{header_extractor.id}@{header_extractor.version}",
                            f"{line_extractor.id}@{line_extractor.version}"),
        fields_produced=len(fields), lines_produced=len(lines),
        warnings=len(field_warnings) + len(line_warnings),
        comparison=compare_fields(generic_fields, fields, generic_lines, lines),
    )
    return result


def _diagnose_in_child(send, pages, data, generic_fields, generic_lines):
    try:
        send.send({"status": "running", "detector_version": "as400-template-router@1",
                   "template": "unknown", "abstention": "classification-not-completed",
                   "extractor_versions": (f"{GENERIC.id}@{GENERIC.version}",),
                   "fields_produced": 0, "lines_produced": 0, "warnings": 0})
        result = _diagnose(pages, data, generic_fields, generic_lines,
                           on_classification=lambda classification: send.send(classification | {"status": "running"}))
        send.send(result)
    except Exception as exc:
        send.send({"status": "worker_error", "error_type": type(exc).__name__})
    finally:
        send.close()


def _run_isolated(target, args, timeout):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args), daemon=True)
    try:
        process.start()
        sender.close()
        process.join(timeout)
        last_result = None
        while receiver.poll():
            try:
                last_result = receiver.recv()
            except EOFError:
                break
        if process.is_alive():
            process.terminate()
            process.join()
            return {**(last_result or {}), "status": "timeout", "error_type": "TimeoutError"}
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
        timeout=TIMEOUT_SECONDS):
    """Run in a killable child so a slow PDF/parser cannot hold the worker indefinitely."""
    if not enabled() or not attempt_id or not _still_current(document_id, attempt_id, db_path):
        return None

    started = time.monotonic()
    result = None
    result = _run_isolated(_diagnose_in_child, (pages, data, generic_fields, generic_lines), timeout)

    if not _still_current(document_id, attempt_id, db_path):
        return None
    result.setdefault("detector_version", "as400-template-router@1")
    result.setdefault("template", "unknown")
    result.setdefault("abstention", "classification-not-completed")
    result.setdefault("extractor_versions", (f"{GENERIC.id}@{GENERIC.version}",))
    result.setdefault("fields_produced", 0)
    result.setdefault("lines_produced", 0)
    result.setdefault("warnings", 0)
    result.update(document_id=document_id, attempt_id=attempt_id,
                  duration_ms=round((time.monotonic() - started) * 1000))
    LOGGER.info("AS/400 shadow extraction diagnostic", extra={"as400_shadow": result})
    return result
