"""Offline evaluation of registered invoice-template detectors."""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

from application.template_detection import detect_template
from application.extractor_registry import template_line_extractor_for
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


LINE_NUMERIC_FIELDS = {"quantity", "unit_price", "amount", "vat_rate", "vat_amount", "total",
                       "documentary_total", "t_cheques", "reference"}
LINE_DECIMAL_FIELDS = {"quantity", "unit_price", "amount", "vat_rate", "vat_amount", "total",
                       "documentary_total"}
LINE_FIELD_ALIASES = {"t_cheques_unverified": "t_cheques"}
LINE_CLASSES = ("charge", "discount", "informational")
ROW_MATCH_Y_TOLERANCE = 0.009


def _line_value(field, value):
    if value is None:
        return None
    if field == "line_date":
        if isinstance(value, date):
            return value.isoformat()
        for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(value), pattern).date().isoformat()
            except ValueError:
                pass
    if field in {"reference", "t_cheques"}:
        return format(value, "f") if isinstance(value, Decimal) else str(value).strip()
    if field in LINE_NUMERIC_FIELDS or isinstance(value, Decimal):
        try:
            raw = str(value).strip().replace("€", "").replace("EUR", "").replace(" ", "")
            number = Decimal(raw.replace(".", "").replace(",", ".") if "," in raw else raw)
            return number
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value)
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return "".join(re.findall(r"[A-Z0-9]+", normalized.upper()))
    return value


def _line_rect(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _gold_line_rect(line):
    return _line_rect(field.get("bbox") for field in line["fields"].values()
                      if field["status"] == "confirmed")


def _line_view(line):
    if isinstance(line, dict):
        return line
    return {"description": line.description, "values": line.values, "evidence": line.evidence}


def _predicted_line_rect(line):
    line = _line_view(line)
    return _line_rect((e.bounds.x0, e.bounds.top, e.bounds.x1, e.bounds.bottom)
                      for e in line["evidence"] if e.bounds is not None)


def _line_pair_score(gold, predicted, page_sizes):
    predicted = _line_view(predicted)
    gold_rect, predicted_rect = _gold_line_rect(gold), _predicted_line_rect(predicted)
    if not gold_rect or not predicted_rect:
        return None
    gold_pages = {gold["page"]}
    predicted_pages = {e.page_number for e in predicted["evidence"] if e.bounds is not None}
    common_pages = gold_pages & predicted_pages
    if len(common_pages) != 1:
        return None
    page = next(iter(common_pages))
    _, page_height = page_sizes[page]
    gy = (gold_rect[1] + gold_rect[3]) / 2
    py = (predicted_rect[1] + predicted_rect[3]) / 2
    dy = abs(gy - py) / page_height
    if dy > ROW_MATCH_Y_TOLERANCE:
        return None
    overlap_x = max(0, min(gold_rect[2], predicted_rect[2]) - max(gold_rect[0], predicted_rect[0]))
    min_width = min(gold_rect[2] - gold_rect[0], predicted_rect[2] - predicted_rect[0])
    x_coverage = overlap_x / min_width if min_width > 0 else 0
    if x_coverage < 0.25:
        return None
    return 1 - dy / ROW_MATCH_Y_TOLERANCE + x_coverage


def _match_line_rows(gold_rows, predicted_rows, page_sizes):
    """Maximize geometric one-to-one matches; values and classes never form edges."""
    scores = [[_line_pair_score(g, p, page_sizes) for p in predicted_rows] for g in gold_rows]
    # ponytail: exponential bitmask DP fits few-row invoices; switch to Hungarian if line counts grow.
    @lru_cache(None)
    def best(gold_index, used):
        if gold_index == len(gold_rows):
            return (0, 0.0, ())
        count, total, assignment = best(gold_index + 1, used)
        winner = (count, total, (len(predicted_rows),) + assignment)
        for predicted_index, score in enumerate(scores[gold_index]):
            if score is None or used & (1 << predicted_index):
                continue
            tail_count, tail_total, tail_assignment = best(gold_index + 1, used | (1 << predicted_index))
            candidate = (tail_count + 1, tail_total + score, (predicted_index,) + tail_assignment)
            if (candidate[0], round(candidate[1], 12)) > (winner[0], round(winner[1], 12)):
                winner = candidate
            elif (candidate[0], round(candidate[1], 12)) == (winner[0], round(winner[1], 12)):
                if candidate[2] < winner[2]:
                    winner = candidate
        return winner

    _, _, assignment = best(0, 0)
    pairs = [(i, j) for i, j in enumerate(assignment) if j < len(predicted_rows)]
    matched_gold, matched_predictions = {i for i, _ in pairs}, {j for _, j in pairs}
    unmatched_gold = [i for i in range(len(gold_rows)) if i not in matched_gold]
    unmatched_predictions = [j for j in range(len(predicted_rows)) if j not in matched_predictions]
    return pairs, unmatched_gold, unmatched_predictions, scores


def _evidence_in_gold_cell(predicted_line, field):
    page_number = field["page"]
    x0, top, x1, bottom = field["bbox"]
    tolerance = 0.05
    return [e for e in predicted_line["evidence"] if e.page_number == page_number and e.bounds is not None
            and e.bounds.x1 >= x0 - tolerance and e.bounds.x0 <= x1 + tolerance
            and e.bounds.bottom >= top - tolerance and e.bounds.top <= bottom + tolerance]


def _source_quote(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip().casefold()


def _field_evidence_score(predicted_line, field, *, column_evidence_evaluable):
    page_number = field["page"]
    evidence = predicted_line["evidence"]
    page_correct = any(e.page_number == field["page"] for e in evidence)
    row_rect = _predicted_line_rect(predicted_line)
    gold_rect = _gold_line_rect({"fields": {"field": field}})
    row_associated = bool(row_rect and gold_rect and any(
        e.page_number == field["page"] and e.bounds is not None
        and abs((e.bounds.top + e.bounds.bottom - gold_rect[1] - gold_rect[3]) / 2) <= 5
        for e in evidence))
    cell_evidence = _evidence_in_gold_cell(predicted_line, field)
    tokens_supported = any(e.source_token_ids for e in evidence)
    predicted_rect = _line_rect((e.bounds.x0, e.bounds.top, e.bounds.x1, e.bounds.bottom)
                                for e in cell_evidence)
    gold_box = tuple(field["bbox"])
    bbox_correct = bool(predicted_rect and all(abs(a - b) <= 0.02
                                                for a, b in zip(predicted_rect, gold_box)))
    if column_evidence_evaluable and not cell_evidence and any(e.page_number == page_number and e.bounds
                                                               for e in evidence):
        bbox_correct = False
    predicted_quote = " ".join(e.quote for e in sorted(cell_evidence,
                                                       key=lambda e: (e.bounds.top, e.bounds.x0)))
    source_tokens_correct = bool(cell_evidence and tokens_supported
                                 and _source_quote(predicted_quote) == _source_quote(field.get("quote"))
                                 and bbox_correct)
    column_associated = bool(cell_evidence and source_tokens_correct) if column_evidence_evaluable else None
    return {"page": page_correct, "row": row_associated, "column": column_associated,
            "source_tokens": (source_tokens_correct if column_evidence_evaluable and tokens_supported else None),
            "bbox": bbox_correct if column_evidence_evaluable and any(e.page_number == page_number
                                                                       and e.bounds for e in evidence) else None}


def _empty_field_score():
    return {"support": 0, "correct": 0, "omitted": 0, "wrong_values": 0,
            "invented_absent": 0, "misassigned": 0, "accuracy": None}


def _score_line_document(case_id, template, gold_rows, predicted_rows, page_sizes, engine):
    predicted_rows = [_line_view(line) for line in predicted_rows]
    pairs, unmatched_gold, unmatched_predictions, scores = _match_line_rows(gold_rows, predicted_rows, page_sizes)
    paired = dict(pairs)
    by_column = defaultdict(_empty_field_score)
    numeric = {"support": 0, "exact": 0, "sign_support": 0, "sign_correct": 0}
    evidence = {key: {"support": 0, "correct": 0, "evaluable": True}
                for key in ("page", "row", "column", "source_tokens", "bbox")}
    if engine != "as400-specialized":
        for key in ("column", "source_tokens", "bbox"):
            evidence[key]["evaluable"] = False
    classification_confusion = {actual: {pred: 0 for pred in (*LINE_CLASSES, "unknown", "missed")}
                                for actual in LINE_CLASSES}
    row_errors = []
    classification_correct = 0
    informational_to_charge = discounts_missed = possible_discount_fusions = 0
    for gold_index, gold in enumerate(gold_rows):
        prediction_index = paired.get(gold_index)
        predicted = predicted_rows[prediction_index] if prediction_index is not None else None
        gold_class = gold["classification"]
        class_evaluable = engine == "as400-specialized"
        predicted_class = ((predicted["values"].get("row_type") or "unknown")
                           if predicted is not None else "missed")
        if predicted_class not in LINE_CLASSES:
            predicted_class = "unknown"
        if class_evaluable:
            classification_confusion[gold_class][predicted_class] += 1
        if class_evaluable and predicted is not None and predicted_class == gold_class:
            classification_correct += 1
        elif class_evaluable and predicted is not None:
            row_errors.append({"type": "classification_error", "gold_row": gold["row_order"],
                               "prediction_row": predicted.get("values", {}).get("row_order")})
            if gold_class == "informational" and predicted_class == "charge":
                informational_to_charge += 1
        if prediction_index is None:
            row_errors.append({"type": "omitted_row", "gold_row": gold["row_order"]})
            if gold_class == "discount":
                discounts_missed += 1
        actual_values = {}
        if predicted is not None:
            actual_values = {LINE_FIELD_ALIASES.get(name, name): value
                             for name, value in predicted["values"].items()
                             if name not in {"row_type", "row_order"}}
            if predicted.get("description") is not None:
                actual_values["description"] = predicted["description"]
        for name, source_reference in gold["fields"].items():
            reference = {**source_reference, "page": gold["page"]}
            score = by_column[name]
            status = reference["status"]
            value = actual_values.get(name) if predicted is not None else None
            has_value = value is not None
            if status == "absent":
                if predicted is None:
                    continue
                score["support"] += 1
                if has_value:
                    score["invented_absent"] += 1
                    row_errors.append({"type": "invented_absent_field", "gold_row": gold["row_order"],
                                       "column": name})
                else:
                    score["correct"] += 1
                continue
            if status != "confirmed":
                continue
            score["support"] += 1
            expected = _line_value(name, reference["value"])
            actual = _line_value(name, value)
            exact = has_value and actual == expected
            if exact:
                score["correct"] += 1
            elif not has_value:
                score["omitted"] += 1
                row_errors.append({"type": "omitted_field", "gold_row": gold["row_order"], "column": name})
            else:
                score["wrong_values"] += 1
                row_errors.append({"type": "wrong_field_value", "gold_row": gold["row_order"], "column": name})
            other_key = next((other for other, candidate in actual_values.items()
                              if other != name and _line_value(name, candidate) == expected), None)
            if other_key and not exact:
                score["misassigned"] += 1
                row_errors.append({"type": "column_misassignment", "gold_row": gold["row_order"],
                                   "column": name})
            if name in LINE_DECIMAL_FIELDS:
                numeric["support"] += 1
                numeric["exact"] += bool(exact)
                numeric["sign_support"] += 1
                if has_value and isinstance(actual, Decimal) and isinstance(expected, Decimal):
                    same_sign = (actual > 0) - (actual < 0) == (expected > 0) - (expected < 0)
                    numeric["sign_correct"] += same_sign
                    if not same_sign:
                        row_errors.append({"type": "sign_error", "gold_row": gold["row_order"],
                                           "column": name})
            if predicted is not None:
                ev = _field_evidence_score(predicted, reference, column_evidence_evaluable=
                                           engine == "as400-specialized")
                for metric, correct in ev.items():
                    if correct is None:
                        evidence[metric]["evaluable"] = False
                        continue
                    evidence[metric]["support"] += 1
                    evidence[metric]["correct"] += bool(correct)
                    if metric in {"source_tokens", "bbox", "column"} and not correct:
                        row_errors.append({"type": f"evidence_{metric}_error", "gold_row": gold["row_order"],
                                           "column": name})
    for prediction_index in unmatched_predictions:
        predicted = predicted_rows[prediction_index]
        candidates = [i for i, score in enumerate(scores) if score[prediction_index] is not None]
        error_type = "duplicate_prediction" if candidates else "additional_row"
        row_errors.append({"type": error_type,
                           "prediction_row": predicted.get("values", {}).get("row_order")})
    for prediction in predicted_rows:
        pred_rect = _predicted_line_rect(prediction)
        if pred_rect:
            overlaps = [row for row in gold_rows if row["classification"] == "discount"
                        and _gold_line_rect(row) and row["page"] == prediction["evidence"][0].page_number
                        and pred_rect[1] < _gold_line_rect(row)[3]
                        and pred_rect[3] > _gold_line_rect(row)[1]]
            possible_discount_fusions += len(overlaps) > 1
    for column, score in by_column.items():
        score["accuracy"] = score["correct"] / score["support"] if score["support"] else None
    detection_tp = len(pairs)
    detection_fp, detection_fn = len(unmatched_predictions), len(unmatched_gold)
    precision = detection_tp / (detection_tp + detection_fp) if detection_tp + detection_fp else None
    recall = detection_tp / (detection_tp + detection_fn) if detection_tp + detection_fn else None
    detection = {"gold_expected": len(gold_rows), "predictions_emitted": len(predicted_rows),
                 "true_positives": detection_tp, "false_positives": detection_fp,
                 "false_negatives": detection_fn, "precision": precision, "recall": recall,
                 "f1": 2 * precision * recall / (precision + recall) if precision and recall else
                 (0.0 if precision == 0 or recall == 0 else None)}
    classification_support = sum(sum(row.values()) for row in classification_confusion.values())
    class_accuracy = classification_correct / classification_support if classification_support else None
    by_class = {kind: {"support": sum(classification_confusion[kind].values()),
                       "correct": classification_confusion[kind][kind],
                       "accuracy": (classification_confusion[kind][kind]
                                   / sum(classification_confusion[kind].values())
                                   if sum(classification_confusion[kind].values()) else None)}
                for kind in LINE_CLASSES}
    return {"case_id": case_id, "template": template, "engine": engine, "detection": detection,
            "classification": {"accuracy": class_accuracy, "support": classification_support,
                               "evaluable": engine == "as400-specialized",
                               "by_class": by_class,
                               "confusion": classification_confusion,
                               "discounts_omitted": discounts_missed,
                               "possible_discount_fusions": possible_discount_fusions,
                               "informational_as_charge": informational_to_charge},
            "fields": {"by_column": dict(by_column),
                       "decimal": {**numeric, "accuracy": numeric["exact"] / numeric["support"]
                                   if numeric["support"] else None,
                                   "sign_accuracy": numeric["sign_correct"] / numeric["sign_support"]
                                   if numeric["sign_support"] else None}},
            "evidence": {key: {**value, "accuracy": value["correct"] / value["support"]
                               if value["evaluable"] and value["support"] else None}
                         for key, value in evidence.items()},
            "unmatched_gold_rows": [gold_rows[i]["row_order"] for i in unmatched_gold],
            "unmatched_prediction_rows": [predicted_rows[i].get("values", {}).get("row_order")
                                          for i in unmatched_predictions],
            "diagnostics": row_errors}


def _score_document_notes(gold_notes, predicted_notes, page_sizes):
    gold_classes = {note["classification"] for note in gold_notes}
    predicted_classes = Counter(line.values.get("row_type", "unknown") for line in predicted_notes)
    by_class = {}
    for classification in sorted(gold_classes):
        gold_rows = []
        for order, note in enumerate((n for n in gold_notes if n["classification"] == classification), 1):
            gold_rows.append({"page": note["page"], "row_order": order, "classification": classification,
                              "fields": {"note": {"status": "confirmed", "page": note["page"],
                                                   "value": note["value"], "quote": note["quote"],
                                                   "tokens": note["tokens"], "bbox": note["bbox"]}}})
        notes = [line for line in predicted_notes if line.values.get("row_type") == classification]
        predicted_rows = [{"description": line.description, "values": line.values, "evidence": line.evidence}
                          for line in notes]
        pairs, missing, extra, _ = _match_line_rows(gold_rows, predicted_rows, page_sizes)
        boxes_correct = quotes_correct = 0
        for gold_index, prediction_index in pairs:
            field = {**gold_rows[gold_index]["fields"]["note"], "page": gold_rows[gold_index]["page"]}
            evidence = _evidence_in_gold_cell(predicted_rows[prediction_index], field)
            box = _line_rect((e.bounds.x0, e.bounds.top, e.bounds.x1, e.bounds.bottom) for e in evidence)
            boxes_correct += bool(box and all(abs(a - b) <= 0.02 for a, b in zip(box, field["bbox"])))
            quote = " ".join(e.quote for e in sorted(evidence, key=lambda e: (e.bounds.top, e.bounds.x0)))
            quotes_correct += bool(evidence and _source_quote(quote) == _source_quote(field["quote"]))
        by_class[classification] = {"gold": len(gold_rows), "predicted": len(notes), "matched": len(pairs),
                                    "false_positives": len(extra), "false_negatives": len(missing),
                                    "evidence_bbox_correct": boxes_correct,
                                    "source_quote_correct": quotes_correct}
    unassessed = sum(count for classification, count in predicted_classes.items() if classification not in gold_classes)
    return {"by_class": by_class, "predicted_by_classification": dict(predicted_classes),
            "unassessed_predictions": unassessed, "evaluable": bool(gold_classes)}


def evaluate_template_lines(line_manifest_path, pdf_root, inventory_path, template_manifest_path):
    line_manifest = json.loads(Path(line_manifest_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    template_manifest = json.loads(Path(template_manifest_path).read_text())
    template_by_case = {case["case_id"]: case["template"] for case in template_manifest["cases"]}
    inventory_by_case = {doc["case_id"]: doc for doc in inventory["documents"]}
    root = Path(pdf_root).resolve()
    results = defaultdict(list)
    selector = {"expected_unknown_documents": 0, "unknown_abstained_as_expected": 0,
                "known_documents": 0, "known_documents_selected": 0}
    diagnostics = []
    note_metrics = []
    document_note_predictions = Counter()
    for case_id, gold_rows in line_manifest["documents"].items():
        template = template_by_case[case_id]
        source_name = inventory_by_case[case_id]["source_name"]
        pdf_path = (root / source_name).resolve()
        if root not in pdf_path.parents or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path escapes authorized root for {case_id}")
        data = pdf_path.read_bytes()
        pages = extract_pdf(data)
        page_sizes = {page.number: (page.width, page.height) for page in pages}
        detector = detect_template(pages)
        gold_document_notes = line_manifest.get("document_notes", {}).get(case_id, [])
        if template == "unknown":
            selector["expected_unknown_documents"] += 1
            selector["unknown_abstained_as_expected"] += detector.template == "unknown"
            specialized_lines = ()
            specialized_notes = ()
            specialized_warnings = ("expected-unknown-template",)
            specialized_called = False
        else:
            selector["known_documents"] += 1
            selector["known_documents_selected"] += detector.template in {"T01", "T02"}
            if detector.template in {"T01", "T02"}:
                specialized_called = True
                _, specialized_lines, specialized_warnings = template_line_extractor_for(
                    detector.template).extract(pages, extract_pdf_tokens(data))
                specialized_notes = tuple(line for line in specialized_lines
                                          if line.values.get("document_note"))
                specialized_lines = tuple(line for line in specialized_lines
                                          if not line.values.get("document_note"))
            else:
                specialized_called = False
                specialized_lines, specialized_notes = (), ()
                specialized_warnings = (detector.reason or "template-abstained",)
        _, generic_lines, _ = extract_invoice(pages)
        specialized_note_metrics = _score_document_notes(gold_document_notes, specialized_notes, page_sizes)
        note_types = Counter(line.values.get("row_type", "unknown") for line in specialized_notes)
        document_note_predictions.update(note_types)
        for engine, predictions in (("as400-specialized", specialized_lines),
                                    ("generic-invoice@1", generic_lines)):
            scored_gold = [] if engine == "as400-specialized" and template == "unknown" else gold_rows
            result = _score_line_document(case_id, template, scored_gold, predictions, page_sizes, engine)
            result["detector_template"] = detector.template
            result["warnings"] = list(specialized_warnings) if engine == "as400-specialized" else []
            result["document_notes"] = (specialized_note_metrics if engine == "as400-specialized" else
                {"gold": 0, "predicted": 0, "matched": 0, "false_positives": 0,
                 "false_negatives": 0, "evidence_bbox_correct": 0, "source_quote_correct": 0,
                 "evaluable": False})
            results[(template, engine)].append(result)
        diagnostics.append({"case_id": case_id, "gold_template": template,
                            "detector_template": detector.template,
                            "specialized_extractor_used": specialized_called,
                            "document_notes_by_classification": dict(note_types),
                            "specialized_warnings": list(specialized_warnings),
                            "document_notes": specialized_note_metrics})

    note_docs = [doc for rows in results.values() for doc in rows
                 if doc["engine"] == "as400-specialized"]
    note_classes = sorted({kind for doc in note_docs for kind in doc["document_notes"]["by_class"]})
    note_metrics.append({"engine": "as400-specialized",
                         "by_class": {kind: {key: sum(doc["document_notes"]["by_class"].get(kind, {}).get(key, 0)
                                                       for doc in note_docs)
                                              for key in ("gold", "predicted", "matched", "false_positives",
                                                          "false_negatives", "evidence_bbox_correct",
                                                          "source_quote_correct")}
                                      for kind in note_classes},
                         "unassessed_predictions": sum(doc["document_notes"]["unassessed_predictions"]
                                                        for doc in note_docs),
                         "evaluable": bool(note_classes)})
    note_metrics.append({"engine": "generic-invoice@1", "by_class": {},
                         "unassessed_predictions": None, "evaluable": False})

    def aggregate(template, engine):
        docs = [doc for (doc_template, doc_engine), values in results.items()
                if doc_engine == engine and (template == "all" or doc_template == template)
                and (engine != "as400-specialized" or doc_template != "unknown")
                for doc in values]
        metrics = {}
        for section in ("detection", "classification"):
            if section == "detection":
                keys = ("gold_expected", "predictions_emitted", "true_positives", "false_positives", "false_negatives")
                total = {key: sum(doc[section][key] for doc in docs) for key in keys}
                tp, fp, fn = total["true_positives"], total["false_positives"], total["false_negatives"]
                p = tp / (tp + fp) if tp + fp else None
                r = tp / (tp + fn) if tp + fn else None
                metrics[section] = {**total, "precision": p, "recall": r,
                                    "f1": 2 * p * r / (p + r) if p and r else (0.0 if p == 0 or r == 0 else None)}
            else:
                confusion = {actual: {pred: sum(d[section]["confusion"][actual][pred] for d in docs)
                                      for pred in (*LINE_CLASSES, "unknown", "missed")} for actual in LINE_CLASSES}
                support = sum(sum(row.values()) for row in confusion.values())
                correct = sum(confusion[k][k] for k in LINE_CLASSES)
                by_class = {kind: {"support": sum(confusion[kind].values()),
                                   "correct": confusion[kind][kind],
                                   "accuracy": confusion[kind][kind] / sum(confusion[kind].values())
                                   if sum(confusion[kind].values()) else None}
                            for kind in LINE_CLASSES}
                metrics[section] = {"accuracy": correct / support if support else None, "support": support,
                                    "evaluable": engine == "as400-specialized",
                                    "by_class": by_class,
                                    "confusion": confusion,
                                    "discounts_omitted": sum(d[section]["discounts_omitted"] for d in docs),
                                    "possible_discount_fusions": sum(d[section]["possible_discount_fusions"] for d in docs),
                                    "informational_as_charge": sum(d[section]["informational_as_charge"] for d in docs)}
        columns = sorted({name for d in docs for name in d["fields"]["by_column"]})
        by_column = {}
        for name in columns:
            rows = [d["fields"]["by_column"][name] for d in docs if name in d["fields"]["by_column"]]
            support = sum(row["support"] for row in rows)
            correct = sum(row["correct"] for row in rows)
            by_column[name] = {"support": support, "correct": correct,
                               "accuracy": correct / support if support else None,
                               "omitted": sum(row["omitted"] for row in rows),
                               "wrong_values": sum(row["wrong_values"] for row in rows),
                               "invented_absent": sum(row["invented_absent"] for row in rows),
                               "misassigned": sum(row["misassigned"] for row in rows)}
        numeric = [d["fields"]["decimal"] for d in docs]
        n_support = sum(row["support"] for row in numeric)
        n_exact = sum(row["exact"] for row in numeric)
        s_support = sum(row["sign_support"] for row in numeric)
        s_correct = sum(row["sign_correct"] for row in numeric)
        metrics["fields"] = {"by_column": by_column,
                             "decimal": {"support": n_support, "exact": n_exact,
                                         "accuracy": n_exact / n_support if n_support else None,
                                         "sign_support": s_support, "sign_correct": s_correct,
                                         "sign_accuracy": s_correct / s_support if s_support else None}}
        metrics["evidence"] = {}
        for metric in ("page", "row", "column", "source_tokens", "bbox"):
            supports = [d["evidence"][metric] for d in docs if d["evidence"][metric]["evaluable"]]
            support = sum(item["support"] for item in supports)
            correct = sum(item["correct"] for item in supports)
            metrics["evidence"][metric] = {"support": support, "correct": correct,
                                           "accuracy": correct / support if support else None,
                                           "evaluable": support > 0}
        metrics["errors"] = dict(Counter(error["type"] for d in docs for error in d["diagnostics"]))
        metrics["documents"] = len(docs)
        return metrics

    by_template = {template: {engine: aggregate(template, engine)
                              for engine in ("as400-specialized", "generic-invoice@1")}
                   for template in ("T01", "T02")}
    all_gold_rows = [row for rows in line_manifest["documents"].values() for row in rows]
    corpus = {"gold_rows": len(all_gold_rows), "known_template_rows": sum(
        len(rows) for case_id, rows in line_manifest["documents"].items()
        if template_by_case[case_id] in {"T01", "T02"}),
        "unknown_document_rows": sum(len(rows) for case_id, rows in line_manifest["documents"].items()
                                      if template_by_case[case_id] == "unknown"),
        "selector": selector}
    all_metrics = {}
    for engine in ("as400-specialized", "generic-invoice@1"):
        combined = [doc for (template, name), docs in results.items()
                    if name == engine and (engine != "as400-specialized" or template != "unknown")
                    for doc in docs]
        gold_rows = sum(d["detection"]["gold_expected"] for d in combined)
        tp = sum(d["detection"]["true_positives"] for d in combined)
        fp = sum(d["detection"]["false_positives"] for d in combined)
        fn = sum(d["detection"]["false_negatives"] for d in combined)
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        all_metrics[engine] = {"gold_expected": gold_rows, "predictions_emitted": sum(
            d["detection"]["predictions_emitted"] for d in combined), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p and r else (0.0 if p == 0 or r == 0 else None)}
    return {"version": 1, "matching_rule": {"basis": "same page, vertical row-center distance <= 0.009 page heights, horizontal interval overlap >= 0.25; exact maximum-cardinality then maximum-geometric-score one-to-one assignment; deterministic index tie-break; values, descriptions and classes excluded",
                                               "unknown_policy": "no specialized extractor is called when the selector returns unknown"},
            "by_template": by_template,
            "document_note_metrics": note_metrics,
            "corpus_metrics": {engine: aggregate("all", engine)
                               for engine in ("as400-specialized", "generic-invoice@1")},
            "corpus_coverage": corpus, "corpus_detection": all_metrics,
            "document_note_predictions_by_classification": dict(document_note_predictions),
            "document_diagnostics": diagnostics,
            "document_results": [d for rows in results.values() for d in rows],
            "private_values_quotes_filenames_and_paths_omitted": True}


def markdown_line_report(report):
    def pct(value):
        return "n/a" if value is None else f"{value:.3f}"
    lines = ["# Line Extraction Evaluation", "", "Matching: " + report["matching_rule"]["basis"], "",
             "## Detection", "", "| Template | Engine | Gold | Predicted | TP | FP | FN | Precision | Recall | F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            d = metrics["detection"]
            lines.append(f"| {template} | {engine} | {d['gold_expected']} | {d['predictions_emitted']} | "
                         f"{d['true_positives']} | {d['false_positives']} | {d['false_negatives']} | "
                         f"{pct(d['precision'])} | {pct(d['recall'])} | {pct(d['f1'])} |")
    lines.extend(["", "## Classification", "", "| Template | Engine | Accuracy | Support | Evaluability | Discounts omitted | Possible discount fusions | Informational as charge |",
                  "|---|---|---:|---:|---|---:|---:|---:|"])
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            c = metrics["classification"]
            lines.append(f"| {template} | {engine} | {pct(c['accuracy'])} | {c['support']} | "
                         f"{'yes' if c['evaluable'] else 'no'} | "
                         f"{c['discounts_omitted']} | {c['possible_discount_fusions']} | {c['informational_as_charge']} |")
    lines.extend(["", "### Class Accuracy", "", "| Template | Engine | Class | Accuracy | Support |",
                  "|---|---|---|---:|---:|"])
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            for kind, score in metrics["classification"]["by_class"].items():
                lines.append(f"| {template} | {engine} | {kind} | {pct(score['accuracy'])} | {score['support']} |")
    lines.extend(["", "## Corpus Comparison", "",
                  "| Engine | Gold rows | Predicted | TP | FP | FN | Precision | Recall | F1 | Decimal accuracy | Sign accuracy |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for engine, metrics in report["corpus_metrics"].items():
        d, decimal = metrics["detection"], metrics["fields"]["decimal"]
        lines.append(f"| {engine} | {d['gold_expected']} | {d['predictions_emitted']} | {d['true_positives']} | "
                     f"{d['false_positives']} | {d['false_negatives']} | {pct(d['precision'])} | "
                     f"{pct(d['recall'])} | {pct(d['f1'])} | {pct(decimal['accuracy'])} | "
                     f"{pct(decimal['sign_accuracy'])} |")
    lines.extend(["", "### Global Field Accuracy", "",
                  "| Engine | Column | Exact / support | Accuracy | Omitted | Wrong | Invented absent | Misassigned |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for engine, metrics in report["corpus_metrics"].items():
        for column, score in metrics["fields"]["by_column"].items():
            lines.append(f"| {engine} | {column} | {score['correct']}/{score['support']} | {pct(score['accuracy'])} | "
                         f"{score['omitted']} | {score['wrong_values']} | {score['invented_absent']} | {score['misassigned']} |")
    lines.extend(["", "### Global Classification", "", "| Engine | Class | Accuracy | Support | Evaluability |",
                  "|---|---|---:|---:|---|"])
    for engine, metrics in report["corpus_metrics"].items():
        for kind, score in metrics["classification"]["by_class"].items():
            lines.append(f"| {engine} | {kind} | {pct(score['accuracy'])} | {score['support']} | "
                         f"{'yes' if metrics['classification']['evaluable'] else 'no'} |")
    lines.extend(["", "### Global Evidence", "",
                  "| Engine | Evidence | Correct / support | Accuracy | Evaluability |",
                  "|---|---|---:|---:|---|"])
    for engine, metrics in report["corpus_metrics"].items():
        for name, score in metrics["evidence"].items():
            lines.append(f"| {engine} | {name} | {score['correct']}/{score['support']} | "
                         f"{pct(score['accuracy'])} | {'yes' if score['evaluable'] else 'no'} |")
    lines.extend(["", "## Fields", "", "| Template | Engine | Column | Accuracy | Support | Omitted | Wrong | Invented absent | Misassigned |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            for column, score in metrics["fields"]["by_column"].items():
                lines.append(f"| {template} | {engine} | {column} | {pct(score['accuracy'])} | {score['support']} | "
                             f"{score['omitted']} | {score['wrong_values']} | {score['invented_absent']} | {score['misassigned']} |")
            decimal = metrics["fields"]["decimal"]
            lines.append(f"| {template} | {engine} | Decimal values | {pct(decimal['accuracy'])} | {decimal['support']} | - | - | - | - |")
            lines.append(f"| {template} | {engine} | Decimal signs | {pct(decimal['sign_accuracy'])} | {decimal['sign_support']} | - | - | - | - |")
    lines.extend(["", "## Evidence", "", "| Template | Engine | Evidence | Accuracy | Support | Evaluability |",
                  "|---|---|---|---:|---:|---|"])
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            for name, score in metrics["evidence"].items():
                lines.append(f"| {template} | {engine} | {name} | {pct(score['accuracy'])} | "
                             f"{score['support']} | {'yes' if score['evaluable'] else 'no'} |")
    lines.extend(["", "## Error Categories", "", "| Template | Engine | Error type | Count |", "|---|---|---|---:|"])
    for template, engines in report["by_template"].items():
        for engine, metrics in engines.items():
            for error_type, count in sorted(metrics["errors"].items()):
                lines.append(f"| {template} | {engine} | {error_type} | {count} |")
    lines.extend(["", "## Document Notes", "",
                  "| Engine | Classification | Gold | Predicted | Matched | FP | FN | Exact boxes | Source quotes |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for notes in report["document_note_metrics"]:
        for classification, score in notes["by_class"].items():
            lines.append(f"| {notes['engine']} | {classification} | {score['gold']} | {score['predicted']} | "
                         f"{score['matched']} | {score['false_positives']} | {score['false_negatives']} | "
                         f"{score['evidence_bbox_correct']} | {score['source_quote_correct']} |")
        if not notes["evaluable"]:
            lines.append(f"| {notes['engine']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append(f"Unassessed note predictions for {notes['engine']}: {notes['unassessed_predictions']}.")
    selector = report["corpus_coverage"]["selector"]
    lines.extend(["", "## Coverage And Diagnostics", "",
                  f"Corpus rows: {report['corpus_coverage']['gold_rows']}; known-template rows: "
                  f"{report['corpus_coverage']['known_template_rows']}; unknown-template rows: "
                  f"{report['corpus_coverage']['unknown_document_rows']}.",
                  f"Unknown documents abstained as expected: {selector['unknown_abstained_as_expected']}/"
                  f"{selector['expected_unknown_documents']}; known documents selected: "
                  f"{selector['known_documents_selected']}/{selector['known_documents']}.",
                  "Per-document errors are reported by opaque ID and error type only.", ""])
    return "\n".join(lines)


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
