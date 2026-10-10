"""Offline scoring for the generic structured invoice extractor."""

import json
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from domain.invoice_extraction import extract_invoice
from infrastructure.pdf_extraction import extract_pdf


def normalize(field, value):
    if value is None:
        return None
    if field.endswith("date") or field == "issue_date":
        if isinstance(value, date):
            return value.isoformat()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(str(value), fmt).date().isoformat()
            except ValueError:
                pass
    if isinstance(value, Decimal) or field in {"tax_base", "vat", "total", "amount", "unit_price", "quantity"}:
        try:
            raw = str(value).strip().replace("€", "").replace("EUR", "").replace(" ", "")
            number = Decimal(raw.replace(".", "").replace(",", ".") if "," in raw else raw)
            return format(number.normalize(), "f")
        except (InvalidOperation, ValueError):
            pass
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _line_key(line):
    return (
        normalize("description", line.get("description")),
        tuple(sorted((name, normalize(name, value)) for name, value in line.get("values", {}).items())),
    )


def compare_case(case, predicted_fields, predicted_lines):
    expected = case["expected_fields"]
    actual = {field.name: field for field in predicted_fields}
    errors, field_scores = [], {}
    tp = fp = fn = correct = 0
    names = set(expected) | set(actual)
    for name in sorted(names):
        expected_value = expected.get(name)
        predicted = actual.get(name)
        predicted_value = predicted.value if predicted else None
        match = normalize(name, expected_value) == normalize(name, predicted_value)
        if match:
            correct += 1
            if expected_value is not None:
                tp += 1
        elif expected_value is None:
            if predicted_value is not None:
                fp += 1
                errors.append({"type": "invented_field", "field": name})
        elif predicted_value is None:
            fn += 1
            errors.append({"type": "omitted_field", "field": name})
        else:
            tp += 0
            fp += 1
            fn += 1
            assigned_elsewhere = any(
                other != name and other_value is not None
                and normalize(name, predicted_value) == normalize(other, other_value)
                for other, other_value in expected.items()
            )
            errors.append({"type": "misassociated_field" if assigned_elsewhere else "wrong_field_value", "field": name})
        field_scores[name] = {"correct": match, "expected_present": expected_value is not None,
                              "predicted_present": predicted_value is not None}

        refs = case.get("evidence", {}).get(name, [])
        if predicted_value is not None:
            evidence = predicted.evidence if predicted else ()
            correct_evidence = any(
                any(ev.page_number == ref["page"] and normalize("quote", ev.quote) == normalize("quote", ref["quote"])
                    for ev in evidence)
                for ref in refs
            )
            if refs and not correct_evidence:
                errors.append({"type": "incorrect_evidence", "field": name})
            if not refs:
                errors.append({"type": "insufficient_reference_evidence", "field": name})

    expected_lines = case.get("expected_lines", [])
    expected_counts, actual_counts = Counter(map(_line_key, expected_lines)), Counter(map(_line_key, predicted_lines))
    missing = expected_counts - actual_counts
    duplicated = actual_counts - expected_counts
    for _ in range(sum(missing.values())):
        errors.append({"type": "missing_line"})
    for _ in range(sum(duplicated.values())):
        errors.append({"type": "duplicate_or_incorrect_line"})
    # Same description with differing attributes is an association/value error, not two unrelated rows.
    expected_by_description = Counter(line[0] for line in expected_counts.elements())
    actual_by_description = Counter(line[0] for line in actual_counts.elements())
    if expected_by_description == actual_by_description and missing and duplicated:
        misplaced = any(
            actual_key[0] != expected_key[0] and actual_key[1] == expected_key[1]
            for actual_key in actual_counts for expected_key in expected_counts
        )
        errors = [error for error in errors if error["type"] not in {"missing_line", "duplicate_or_incorrect_line"}]
        errors.append({"type": "misassociated_line" if misplaced else "incorrect_line_attributes"})
    expected_by_key = {_line_key(line): line for line in expected_lines}
    predicted_by_key = {_line_key(line): line for line in predicted_lines}
    for key in expected_counts.keys() & actual_counts.keys():
        expected_evidence = expected_by_key[key].get("evidence", [])
        if expected_evidence:
            actual_evidence = predicted_by_key[key].get("evidence", [])
            if not any(any(ev.page_number == ref["page"] and normalize("quote", ev.quote) == normalize("quote", ref["quote"])
                           for ev in actual_evidence) for ref in expected_evidence):
                errors.append({"type": "incorrect_line_evidence"})
        else:
            errors.append({"type": "insufficient_line_reference_evidence"})
    expected_line_count = sum(expected_counts.values())
    line_tp = sum((expected_counts & actual_counts).values())
    line_fp = sum((actual_counts - expected_counts).values())
    line_fn = sum((expected_counts - actual_counts).values())
    return {
        "case_id": case["case_id"], "field_accuracy": correct / len(names) if names else 1.0,
        "fields": field_scores, "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "line_precision": line_tp / (line_tp + line_fp) if line_tp + line_fp else (1.0 if not expected_line_count else 0.0),
        "line_recall": line_tp / expected_line_count if expected_line_count else 1.0,
        "errors": errors,
        "insufficient_evidence": ([name for name in expected if expected[name] is not None and not case.get("evidence", {}).get(name)]
                                  + [f"line:{index}" for index, line in enumerate(expected_lines, 1) if not line.get("evidence")]),
    }


def evaluate(dataset_path):
    path = Path(dataset_path)
    dataset = json.loads(path.read_text())
    if dataset.get("version") != 1:
        raise ValueError("Unsupported extraction dataset version")
    results = []
    for case in dataset["cases"]:
        if not isinstance(case.get("case_id"), str) or not case["case_id"].strip() or not isinstance(case.get("pdf"), str):
            raise ValueError("Every case requires a case_id and PDF path")
        if not isinstance(case.get("expected_fields"), dict) or set(case.get("intentionally_absent", [])) - set(case["expected_fields"]):
            raise ValueError(f"Invalid expected fields in case {case['case_id']}")
        pdf_path = (path.parent / case["pdf"]).resolve()
        if path.parent.resolve() not in pdf_path.parents or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must stay inside the dataset directory in case {case['case_id']}")
        pdf = extract_pdf(pdf_path.read_bytes())
        fields, lines, _ = extract_invoice(pdf)
        results.append(compare_case(case, fields, [
            {"description": line.description, "values": line.values, "evidence": line.evidence} for line in lines
        ]))
    fields = sorted({name for result in results for name in result["fields"]})
    aggregate_fields = {}
    for name in fields:
        rows = [result["fields"][name] for result in results if name in result["fields"]]
        aggregate_fields[name] = {"accuracy": sum(row["correct"] for row in rows) / len(rows),
                                  "support": len(rows)}
    tp = sum(result["true_positive"] for result in results)
    fp = sum(result["false_positive"] for result in results)
    fn = sum(result["false_negative"] for result in results)
    errors = [error for result in results for error in result["errors"]]
    report = {"version": 1, "cases": results,
              "metrics": {"field_accuracy": sum(row["field_accuracy"] for row in results) / len(results) if results else 1.0,
                          "precision": tp / (tp + fp) if tp + fp else 1.0,
                          "recall": tp / (tp + fn) if tp + fn else 1.0,
                          "omitted_fields": sum(error["type"] == "omitted_field" for error in errors),
                          "invented_fields": sum(error["type"] == "invented_field" for error in errors),
                          "wrong_field_values": sum(error["type"] == "wrong_field_value" for error in errors),
                          "misassociated_fields": sum(error["type"] == "misassociated_field" for error in errors),
                          "line_errors": sum(error["type"] in {"missing_line", "duplicate_or_incorrect_line", "misassociated_line"} for error in errors),
                          "evidence_errors": sum("evidence" in error["type"] for error in errors),
                          "cases_insufficient_evidence": sum(bool(result["insufficient_evidence"]) for result in results),
                          "fields": aggregate_fields,
                          "line_precision": sum(row["line_precision"] for row in results) / len(results) if results else 1.0,
                          "line_recall": sum(row["line_recall"] for row in results) / len(results) if results else 1.0}}
    return report


def markdown_report(report):
    metrics = report["metrics"]
    lines = ["# Structured Extraction Evaluation", "", f"Cases: {len(report['cases'])}", "",
             "| Metric | Score |", "|---|---:|",
             f"| Field accuracy | {metrics['field_accuracy']:.3f} |",
             f"| Field precision | {metrics['precision']:.3f} |",
             f"| Field recall | {metrics['recall']:.3f} |",
             f"| Line precision | {metrics['line_precision']:.3f} |",
             f"| Line recall | {metrics['line_recall']:.3f} |", "", "## Per-field accuracy", "",
             "| Field | Accuracy | Support |", "|---|---:|---:|"]
    lines.extend(f"| {name} | {item['accuracy']:.3f} | {item['support']} |" for name, item in metrics["fields"].items())
    lines.extend(["", "## Document errors", ""])
    for case in report["cases"]:
        errors = ", ".join(error["type"] + (":" + error["field"] if "field" in error else "") for error in case["errors"]) or "none"
        lines.append(f"- `{case['case_id']}`: {errors}")
    unresolved = [case["case_id"] for case in report["cases"] if case["insufficient_evidence"]]
    lines.extend(["", "## Cases with insufficient reference evidence", "", ", ".join(f"`{case}`" for case in unresolved) or "none", ""])
    return "\n".join(lines)
