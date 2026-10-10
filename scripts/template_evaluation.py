"""Offline evaluation of registered invoice-template detectors."""

import json
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from application.template_detection import detect_template
from application.template_extraction import extract_classified_header_summary
from domain.invoice_extraction import extract_invoice
from infrastructure.pdf_extraction import extract_pdf, extract_pdf_tokens


CLASSES = ("T01", "T02", "unknown")
BASELINE_ALIASES = {"customer_tax_id": "receiver_tax_id", "vat": "tax_amount", "total": "total_due"}


def _normalize(name, value):
    if value is None:
        return None
    if name == "issue_date":
        if isinstance(value, date):
            return value.isoformat()
        for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(value), pattern).date().isoformat()
            except ValueError:
                pass
    if name in {"tax_base", "tax_amount", "total_invoice", "total_due", "displayed_total_euros"}:
        try:
            raw = str(value).strip().replace("€", "").replace("EUR", "").replace(" ", "")
            number = Decimal(raw.replace(".", "").replace(",", ".") if "," in raw else raw)
            return format(number.normalize(), "f")
        except (InvalidOperation, ValueError):
            return None
    if name == "tax_rates":
        values = value if isinstance(value, (tuple, list)) else (value,)
        normalized = []
        for item in values:
            try:
                normalized.append(format(Decimal(str(item).replace("%", "").replace(",", ".")).normalize(), "f"))
            except InvalidOperation:
                return None
        return tuple(sorted(normalized))
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _refs(field):
    evidence = field.get("evidence")
    if evidence is None:
        return []
    return evidence if isinstance(evidence, list) else [evidence]


def _normalized_bounds(bounds, page):
    return tuple(round(value, 6) for value in (
        bounds.x0 / page.width, bounds.top / page.height,
        bounds.x1 / page.width, bounds.bottom / page.height,
    ))


def _normalize_label(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "".join(re.findall(r"[A-Z0-9]+", text.upper()))


def _evidence_matches(predicted, reference, pages):
    if predicted.page_number != reference.get("page"):
        return False
    page = pages[predicted.page_number - 1]
    if predicted.bounds is None or _normalized_bounds(predicted.bounds, page) != tuple(reference.get("value_bbox", ())):
        return False
    if predicted.quote != reference.get("value_quote"):
        return False
    if reference.get("label_bbox") is not None:
        if predicted.label_bounds is None or _normalized_bounds(predicted.label_bounds, page) != tuple(reference["label_bbox"]):
            return False
        if _normalize_label(predicted.label_quote) != _normalize_label(reference.get("label_quote")):
            return False
    return True


def _aggregate(records):
    confirmed = [item for item in records if item["reference_status"] == "confirmed"]
    tp = sum(item["correct_value"] for item in confirmed)
    fp = sum(item["false_positive"] for item in records)
    fn = sum(item["false_negative"] for item in confirmed)
    covered = sum(item["covered"] for item in confirmed)
    evidence_ref = sum(item["reference_evidence_count"] for item in confirmed)
    evidence_hit = sum(item["correct_evidence_count"] for item in confirmed)
    predicted_evidence = sum(item["predicted_evidence_count"] for item in confirmed)
    return {"support": len(confirmed), "correct": tp,
            "accuracy": tp / len(confirmed) if confirmed else None,
            "coverage": covered / len(confirmed) if confirmed else None,
            "selective_accuracy": tp / covered if covered else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positives": fp, "false_negatives": fn,
            "abstentions": len(confirmed) - covered,
            "evidence_precision": evidence_hit / predicted_evidence if predicted_evidence else None,
            "evidence_recall": evidence_hit / evidence_ref if evidence_ref else None,
            "incorrect_evidence": sum(item["incorrect_evidence"] for item in confirmed),
            "role_errors": sum(item["role_error"] for item in confirmed),
            "amount_association_errors": sum(item["amount_association_error"] for item in confirmed),
            "unresolved_assignments": sum(item["unresolved_assignment"] for item in records),
            "ambiguous_reference_fields": sum(item["reference_status"] == "ambiguous" for item in records)}


def _extraction_metrics(records):
    by_field = {}
    for name in sorted({record["name"] for record in records}):
        by_field[name] = _aggregate([record for record in records if record["name"] == name])
    candidates = [record for record in records if record["reference_status"] == "unverified_semantics"]
    return {"overall": _aggregate(records), "by_field": by_field,
            "documentary_candidates": {"support": len(candidates),
                                       "coverage": sum(record["candidate_covered"] for record in candidates),
                                       "matches": sum(record["candidate_match"] for record in candidates)}}


def _diagnostic_errors(records):
    errors = defaultdict(list)
    for record in records:
        name = record["name"]
        if record["reference_status"] == "confirmed" and not record["covered"]:
            errors[record["case_id"]].append({"field": name, "type": "abstention"})
        elif record["reference_status"] == "confirmed" and not record["correct_value"]:
            errors[record["case_id"]].append({"field": name, "type": "wrong_value"})
        if record["incorrect_evidence"]:
            errors[record["case_id"]].append({"field": name, "type": "incorrect_evidence"})
        if record["role_error"]:
            errors[record["case_id"]].append({"field": name, "type": "issuer_receiver_role_error"})
        if record["amount_association_error"]:
            errors[record["case_id"]].append({"field": name, "type": "amount_association_error"})
        if record["unresolved_assignment"]:
            errors[record["case_id"]].append({"field": name, "type": "ambiguous_reference_assignment"})
    return errors


def _compare_fields(case, pages, predicted_fields, *, baseline=False):
    aliases = BASELINE_ALIASES if baseline else {}
    actual = {aliases.get(item.name, item.name): item for item in predicted_fields}
    fields = []
    for name, reference in case["fields"].items():
        field = actual.get(name)
        status = reference["status"]
        value = field.value if field else None
        prediction_status = (field.status if field else None) or ("confirmed" if value is not None else "absent")
        has_value = value is not None
        evidence = field.evidence if field else ()
        refs = _refs(reference)
        matched_refs = sum(any(_evidence_matches(item, ref, pages) for item in evidence) for ref in refs)
        evidence_ok = bool(refs) and matched_refs == len(refs)
        confirmed_ref = status == "confirmed"
        exact = confirmed_ref and _normalize(name, value) == _normalize(name, reference.get("expected"))
        covered = confirmed_ref and prediction_status == "confirmed" and has_value and bool(evidence)
        false_positive = (status in {"absent", "not_applicable"} and has_value) or (confirmed_ref and has_value and not exact)
        unresolved = status == "ambiguous" and has_value
        amount = name in {"tax_base", "tax_amount", "tax_rates", "total_invoice", "total_due"}
        role_error = False
        if name in {"issuer_tax_id", "receiver_tax_id"} and has_value:
            other = "receiver_tax_id" if name == "issuer_tax_id" else "issuer_tax_id"
            other_value = case["fields"].get(other, {}).get("expected")
            role_error = other_value is not None and _normalize(name, value) == _normalize(other, other_value) and not exact
        evidence_mismatch = confirmed_ref and has_value and bool(refs) and not evidence_ok
        fields.append({"case_id": case["case_id"], "template": case["template"], "name": name,
                       "reference_status": status, "prediction_status": prediction_status,
                       "correct_value": bool(exact), "covered": bool(covered),
                       "false_positive": bool(false_positive), "false_negative": bool(confirmed_ref and not exact),
                       "reference_evidence_count": len(refs) if confirmed_ref else 0,
                       "correct_evidence_count": matched_refs if confirmed_ref else 0,
                       "predicted_evidence_count": len(evidence) if confirmed_ref else 0,
                       "incorrect_evidence": int(evidence_mismatch), "role_error": bool(role_error),
                       "amount_association_error": bool(amount and confirmed_ref and has_value and
                                                        (not exact or not evidence_ok)),
                       "unresolved_assignment": bool(unresolved),
                       "candidate_covered": bool(status == "unverified_semantics" and has_value and bool(evidence)),
                       "candidate_match": bool(status == "unverified_semantics" and has_value and
                                                _normalize(name, value) == _normalize(name, reference.get("expected")))})
    return fields


def evaluate_templates(manifest_path, pdf_root):
    manifest_path = Path(manifest_path).resolve()
    root = Path(pdf_root).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported template evaluation dataset version")

    cases = []
    records_by_template = defaultdict(lambda: defaultdict(list))
    for case in manifest.get("cases", []):
        case_id, expected, relative_pdf = case.get("case_id"), case.get("template"), case.get("pdf")
        line_count = case.get("commercial_line_count")
        if (not isinstance(case_id, str) or expected not in CLASSES or not isinstance(relative_pdf, str)
                or not case.get("reviewed_visually") or not isinstance(line_count, int) or line_count < 0):
            raise ValueError("Each case requires visual review, an opaque case_id, template, line count, and PDF path")
        pdf_path = (root / relative_pdf).resolve()
        if root not in pdf_path.parents or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must stay inside the sample directory in case {case_id}")
        data = pdf_path.read_bytes()
        pages = extract_pdf(data)
        as400 = extract_classified_header_summary(pages, extract_pdf_tokens(data))
        generic_fields, _, _ = extract_invoice(pages)
        selected_fields = _compare_fields(case, pages, as400.fields) if expected != "unknown" else []
        baseline_fields = _compare_fields(case, pages, generic_fields, baseline=True)
        if expected != "unknown":
            records_by_template[expected]["as400"].extend(selected_fields)
        records_by_template[expected]["generic"].extend(baseline_fields)
        cases.append({"case_id": case_id, "expected": expected, "predicted": as400.classification.template,
                      "commercial_line_count": line_count,
                      "visible_detail_rows": case.get("visible_detail_rows"),
                      "business_validation_required": bool(case.get("business_questions")),
                      "detector_id": as400.classification.detector_id,
                      "detector_version": as400.classification.detector_version,
                      "extractor_id": as400.extractor_id, "extractor_version": as400.extractor_version,
                      "signals": as400.classification.signals, "reason": as400.classification.reason,
                      "extraction_errors": {"as400": _diagnostic_errors(selected_fields),
                                            "generic": _diagnostic_errors(baseline_fields)}})

    matrix = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for case in cases:
        matrix[case["expected"]][case["predicted"]] += 1
    accuracy = {}
    for template in CLASSES:
        support = sum(matrix[template].values())
        accuracy[template] = {"correct": matrix[template][template], "support": support,
                              "accuracy": matrix[template][template] / support if support else None}
    per_template = {template: {
        "false_positives": sum(matrix[actual][template] for actual in CLASSES if actual != template),
        "false_negatives": sum(matrix[template][predicted] for predicted in CLASSES if predicted != template),
    } for template in ("T01", "T02")}
    extraction = {}
    for template, engines in records_by_template.items():
        extraction[template] = {engine: _extraction_metrics(records) for engine, records in engines.items()}
    return {"version": 1,
            "classification_rule": "At least 2 signals: exact ordered header plus >=2 fixed labels or matching geometry; one-page portrait ratio 0.68-0.73; multiple matches abstain.",
            "cases": cases,
            "extraction": {"by_template": extraction,
                           "unknown_documents_abstained": sum(c["predicted"] == "unknown" for c in cases),
                           "known_template_abstentions": sum(c["expected"] != "unknown" and c["predicted"] == "unknown" for c in cases),
                           "private_values_and_citations_omitted": True},
            "metrics": {"confusion_matrix": matrix, "accuracy_by_template": accuracy,
                        "false_positives_and_negatives": per_template,
                        "abstentions": sum(case["predicted"] == "unknown" for case in cases),
                        "abstentions_on_known_templates": sum(
                            case["expected"] != "unknown" and case["predicted"] == "unknown" for case in cases),
                        "ambiguous_cases": [case["case_id"] for case in cases
                                            if case["reason"] == "ambiguous-template-match"]}}


def markdown_template_report(report):
    metrics = report["metrics"]
    labels = CLASSES
    lines = ["# Template Detection Evaluation", "",
             "Rule: " + report["classification_rule"], "", "## Confusion matrix", "",
             "| Reference \\ Prediction | " + " | ".join(labels) + " |",
             "|---|---:|---:|---:|"]
    lines.extend("| " + actual + " | " + " | ".join(str(metrics["confusion_matrix"][actual][p]) for p in labels) + " |"
                 for actual in labels)
    lines.extend(["", "## Per-template metrics", "", "| Template | Accuracy | False positives | False negatives |",
                  "|---|---:|---:|---:|"])
    for template in ("T01", "T02", "unknown"):
        score = metrics["accuracy_by_template"][template]
        errors = metrics["false_positives_and_negatives"].get(template, {"false_positives": 0, "false_negatives": 0})
        accuracy = "n/a" if score["accuracy"] is None else f"{score['accuracy']:.3f} ({score['correct']}/{score['support']})"
        lines.append(f"| {template} | {accuracy} | {errors['false_positives']} | {errors['false_negatives']} |")
    lines.extend(["", f"Abstentions: {metrics['abstentions']}",
                  f"Known-template abstentions: {metrics['abstentions_on_known_templates']}",
                  "Ambiguous cases: " + (", ".join(f"`{item}`" for item in metrics["ambiguous_cases"]) or "none"),
                  "", "## Header And Summary Extraction", "",
                  "Only gold fields marked `confirmed` enter exactness metrics. Coverage is reported separately.", "",
                  "| Template | Extractor | Accuracy | Coverage | Selective accuracy | Precision | Recall | Evidence P/R | FP/FN | Abstentions | Role errors | Amount association errors |",
                  "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
    def pct(value):
        return "n/a" if value is None else f"{value:.3f}"
    for template, engines in report["extraction"]["by_template"].items():
        for engine, scores in engines.items():
            metric = scores["overall"]
            evidence = f"{pct(metric['evidence_precision'])}/{pct(metric['evidence_recall'])}"
            lines.append(f"| {template} | {engine} | {pct(metric['accuracy'])} ({metric['correct']}/{metric['support']}) | "
                         f"{pct(metric['coverage'])} | {pct(metric['selective_accuracy'])} | {pct(metric['precision'])} | "
                         f"{pct(metric['recall'])} | {evidence} | {metric['false_positives']}/{metric['false_negatives']} | "
                         f"{metric['abstentions']} | {metric['role_errors']} | {metric['amount_association_errors']} |")
    lines.extend(["", "### Per-field metrics", "", "| Template | Extractor | Field | Correct / support | Accuracy | Coverage | Evidence P/R | FP/FN |",
                  "|---|---|---|---:|---:|---:|---|---:|"])
    for template, engines in report["extraction"]["by_template"].items():
        for engine, scores in engines.items():
            for name, metric in scores["by_field"].items():
                evidence = f"{pct(metric['evidence_precision'])}/{pct(metric['evidence_recall'])}"
                lines.append(f"| {template} | {engine} | {name} | {metric['correct']}/{metric['support']} | "
                             f"{pct(metric['accuracy'])} | {pct(metric['coverage'])} | {evidence} | "
                             f"{metric['false_positives']}/{metric['false_negatives']} |")
    candidates = [(template, engine, scores["documentary_candidates"])
                  for template, engines in report["extraction"]["by_template"].items()
                  for engine, scores in engines.items() if scores["documentary_candidates"]["support"]]
    lines.extend(["", "Documentary candidates with unresolved semantics (excluded from financial accuracy): " +
                  ("; ".join(f"{template}/{engine} {item['matches']}/{item['support']} matched, "
                             f"{item['coverage']} covered" for template, engine, item in candidates) or "none"),
                  f"Documents with no AS/400 extractor selected: {report['extraction']['unknown_documents_abstained']} "
                  f"({report['extraction']['known_template_abstentions']} known-template false abstentions).",
                  "Ambiguous and not-applicable reference fields are excluded from exactness metrics.",
                  "", "## Document diagnostics", "", "| Opaque ID | Reference | Prediction | Extractor | Commercial lines | Business review | Reason | AS400 errors | Generic errors |",
                  "|---|---|---|---:|---|---|---|---|---|"])
    for case in report["cases"]:
        def error_summary(engine):
            errors = case["extraction_errors"][engine]
            return ", ".join(f"{item['field']}:{item['type']}" for items in errors.values() for item in items) or "none"
        lines.append(f"| `{case['case_id']}` | {case['expected']} | {case['predicted']} | "
                     f"{case['extractor_id'] or 'abstained'}@{case['extractor_version'] or '-'} | "
                     f"{case['commercial_line_count']} | {'required' if case['business_validation_required'] else 'no'} | "
                     f"{case['reason'] or 'identified'} | {error_summary('as400')} | {error_summary('generic')} |")
    lines.extend(["", "Sample results only; not an estimate of general system accuracy.", ""])
    return "\n".join(lines)
