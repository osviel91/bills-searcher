import unittest
import json
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from domain.structured_extraction import BoundingBox, Evidence, ExtractedField, PDFPage, ProposedLine
from scripts.template_evaluation import (_aggregate, _compare_fields, _match_line_rows, _normalize,
                                         _score_document_notes, _score_line_document,
                                         evaluate_template_lines)
from application.template_detection import TemplateDetection


class TemplateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.page = PDFPage(1, 1000, 1000, "")

    def test_exact_decimal_and_precise_label_value_boxes_score(self):
        case = {"case_id": "synthetic-01", "template": "T02", "fields": {
            "tax_base": {"status": "confirmed", "expected": "12.34", "evidence": {
                "page": 1, "value_quote": "12,34", "value_bbox": [0.1, 0.2, 0.2, 0.22],
                "label_quote": "BASE IMPONIBLE", "label_bbox": [0.1, 0.16, 0.2, 0.18],
            }},
        }}
        evidence = Evidence(1, "12,34", BoundingBox(100, 200, 200, 220), "p1-w1",
                            "BASE IMPONIBLE", BoundingBox(100, 160, 200, 180), "summary-label-column",
                            "as400-t02-header-summary", "1", ("p1-w1-c0",))
        records = _compare_fields(case, (self.page,), (ExtractedField("tax_base", Decimal("12.34"),
                                                                       (evidence,), status="confirmed"),))
        self.assertTrue(records[0]["correct_value"])
        self.assertEqual(records[0]["correct_evidence_count"], 1)
        self.assertEqual(_aggregate(records)["coverage"], 1.0)
        self.assertNotEqual(_normalize("tax_base", Decimal("12.34")), _normalize("tax_base", Decimal("12.35")))

    def test_wrong_cell_evidence_is_an_association_error_even_when_value_matches(self):
        case = {"case_id": "synthetic-02", "template": "T02", "fields": {
            "tax_base": {"status": "confirmed", "expected": "12.34", "evidence": [{
                "page": 1, "value_quote": "12,34", "value_bbox": [0.1, 0.2, 0.2, 0.22],
                "label_quote": "BASE IMPONIBLE", "label_bbox": [0.1, 0.16, 0.2, 0.18],
            }]},
        }}
        wrong_cell = Evidence(1, "12,34", BoundingBox(300, 200, 400, 220), "p1-w2",
                              "TOTAL FACTURA", BoundingBox(300, 160, 400, 180), "summary-label-column",
                              "as400-t02-header-summary", "1", ("p1-w2-c0",))
        record = _compare_fields(case, (self.page,), (ExtractedField("tax_base", Decimal("12.34"),
                                                                       (wrong_cell,), status="confirmed"),))[0]
        self.assertTrue(record["correct_value"])
        self.assertEqual(record["incorrect_evidence"], 1)
        self.assertTrue(record["amount_association_error"])

    def test_ambiguous_gold_is_excluded_and_assignment_reported_separately(self):
        case = {"case_id": "synthetic-03", "template": "T01", "fields": {
            "total_due": {"status": "ambiguous", "expected": None, "evidence": None},
        }}
        fields = (ExtractedField("total_due", Decimal("12.34"), status="confirmed"),)
        records = _compare_fields(case, (self.page,), fields)
        metrics = _aggregate(records)
        self.assertEqual(metrics["support"], 0)
        self.assertEqual(metrics["unresolved_assignments"], 1)
        self.assertEqual(metrics["false_positives"], 0)

    def test_absent_gold_detects_false_positive(self):
        case = {"case_id": "synthetic-04", "template": "T02", "fields": {
            "tax_base": {"status": "absent", "expected": None, "evidence": None},
        }}
        records = _compare_fields(case, (self.page,), (ExtractedField("tax_base", Decimal("1.00")),))
        self.assertEqual(_aggregate(records)["false_positives"], 1)


class LineMatchingTests(unittest.TestCase):
    page_sizes = {1: (1000, 1000)}

    def gold(self, order, top, description="same", amount="10.00", classification="charge", fields=None):
        fields = fields or {
            "description": {"status": "confirmed", "value": description, "page": 1,
                            "tokens": [f"p1-w{order:04d}"], "bbox": [100, top, 400, top + 10]},
            "amount": {"status": "confirmed", "value": amount, "page": 1,
                       "tokens": [f"p1-w{order + 100:04d}"], "bbox": [800, top, 900, top + 10]},
        }
        return {"page": 1, "row_order": order, "classification": classification, "fields": fields}

    def prediction(self, order, top, description="same", amount="10.00", classification="charge",
                   amount_x=800, amount_quote=None):
        evidence = (
            Evidence(1, description, BoundingBox(100, top, 400, top + 10), source_token_ids=(f"p1-w{order:04d}-c0",)),
            Evidence(1, amount_quote or amount, BoundingBox(amount_x, top, amount_x + 100, top + 10),
                     source_token_ids=(f"p1-w{order + 100:04d}-c0",)),
        )
        return ProposedLine(description, {"amount": Decimal(amount), "row_type": classification,
                                          "row_order": order}, evidence)

    def test_matching_is_one_to_one_and_keeps_identical_occurrences(self):
        gold = [self.gold(1, 100), self.gold(2, 120)]
        predictions = [self.prediction(20, 120), self.prediction(10, 100)]
        pairs, missing, extra, _ = _match_line_rows(gold, predictions, self.page_sizes)
        self.assertEqual(pairs, [(0, 1), (1, 0)])
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])
        self.assertEqual(len({p for _, p in pairs}), len(pairs))

    def test_missing_duplicate_and_additional_rows_are_distinguished(self):
        gold = [self.gold(1, 100), self.gold(2, 120)]
        predictions = [self.prediction(7, 100), self.prediction(8, 100), self.prediction(9, 160)]
        result = _score_line_document("opaque", "T02", gold, predictions, self.page_sizes, "as400-specialized")
        self.assertEqual(result["detection"]["true_positives"], 1)
        self.assertEqual(result["detection"]["false_negatives"], 1)
        self.assertEqual(result["detection"]["false_positives"], 2)
        self.assertTrue({"omitted_row", "duplicate_prediction", "additional_row"} <=
                        {error["type"] for error in result["diagnostics"]})

    def test_values_do_not_create_a_match_for_wrong_row_geometry(self):
        gold = [self.gold(1, 100, amount="99.00")]
        prediction = self.prediction(1, 300, amount="99.00")
        pairs, missing, extra, _ = _match_line_rows(gold, [prediction], self.page_sizes)
        self.assertEqual(pairs, [])
        self.assertEqual(missing, [0])
        self.assertEqual(extra, [0])

    def test_discount_and_informational_class_confusions_are_visible(self):
        gold = [self.gold(1, 100, classification="discount"),
                self.gold(2, 120, classification="informational")]
        predictions = [self.prediction(1, 100, classification="charge"),
                       self.prediction(2, 120, classification="charge")]
        result = _score_line_document("opaque", "T02", gold, predictions, self.page_sizes, "as400-specialized")
        confusion = result["classification"]["confusion"]
        self.assertEqual(confusion["discount"]["charge"], 1)
        self.assertEqual(confusion["informational"]["charge"], 1)
        self.assertEqual(result["classification"]["informational_as_charge"], 1)

    def test_omitted_and_merged_discounts_are_counted(self):
        gold = [self.gold(1, 100, classification="discount"),
                self.gold(2, 120, classification="discount")]
        spanning = ProposedLine("discount", {"row_type": "discount"},
                                (Evidence(1, "row", BoundingBox(100, 95, 900, 125)),))
        result = _score_line_document("opaque", "T02", gold, [spanning], self.page_sizes,
                                      "as400-specialized")
        self.assertEqual(result["classification"]["discounts_omitted"], 1)
        self.assertEqual(result["classification"]["possible_discount_fusions"], 1)

    def test_cancellation_note_evidence_is_scored_separately_from_commercial_rows(self):
        gold_notes = [{"page": 1, "row_order": 1, "classification": "cancellation",
                       "value": "ANULA FACTURA", "quote": "ANULA FACTURA",
                       "tokens": ["p1-w0010"], "bbox": [100, 200, 300, 210]}]
        note = ProposedLine("ANULA FACTURA", {"row_type": "cancellation", "document_note": True},
                            (Evidence(1, "ANULA FACTURA", BoundingBox(100, 200, 300, 210),
                                     source_token_ids=("p1-w10-c0",)),))
        result = _score_document_notes(gold_notes, [note], self.page_sizes)
        score = result["by_class"]["cancellation"]
        self.assertEqual(score["gold"], 1)
        self.assertEqual(score["predicted"], 1)
        self.assertEqual(score["matched"], 1)
        self.assertEqual(score["false_positives"], 0)
        self.assertEqual(score["false_negatives"], 0)
        self.assertEqual(score["evidence_bbox_correct"], 1)
        self.assertEqual(score["source_quote_correct"], 1)

    def test_wrong_column_is_not_rescued_by_a_matching_numeric_value(self):
        fields = {
            "description": {"status": "confirmed", "value": "same", "page": 1,
                            "tokens": ["p1-w0001"], "bbox": [100, 100, 400, 110]},
            "amount": {"status": "confirmed", "value": "10.00", "page": 1,
                       "tokens": ["p1-w0101"], "bbox": [800, 100, 900, 110]},
            "unit_price": {"status": "absent", "value": None, "page": 1, "tokens": [], "bbox": None},
        }
        gold = [self.gold(1, 100, fields=fields)]
        prediction = self.prediction(1, 100, amount_x=500)
        prediction.values["unit_price"] = prediction.values.pop("amount")
        result = _score_line_document("opaque", "T02", gold, [prediction], self.page_sizes, "as400-specialized")
        self.assertEqual(result["fields"]["by_column"]["amount"]["omitted"], 1)
        self.assertEqual(result["fields"]["by_column"]["unit_price"]["invented_absent"], 1)
        self.assertEqual(result["fields"]["by_column"]["amount"]["misassigned"], 1)

    def test_wrong_sign_and_wrong_evidence_box_are_reported(self):
        gold = [self.gold(1, 100, amount="-10.00")]
        prediction = self.prediction(1, 100, amount="10.00", amount_x=500)
        result = _score_line_document("opaque", "T02", gold, [prediction], self.page_sizes, "as400-specialized")
        self.assertEqual(result["fields"]["decimal"]["sign_accuracy"], 0.0)
        self.assertEqual(result["evidence"]["bbox"]["correct"], 1)  # description evidence only
        self.assertTrue(any(error["type"] == "sign_error" for error in result["diagnostics"]))

    def test_generic_line_type_and_precise_cell_evidence_are_not_evaluable(self):
        gold = [self.gold(1, 100)]
        generic = ProposedLine("same", {"amount": Decimal("10.00")},
                               (Evidence(1, "row", BoundingBox(50, 100, 950, 110)),))
        result = _score_line_document("opaque", "T02", gold, [generic], self.page_sizes, "generic-invoice@1")
        self.assertIsNone(result["classification"]["accuracy"])
        self.assertIsNone(result["evidence"]["source_tokens"]["accuracy"])
        self.assertIsNone(result["evidence"]["bbox"]["accuracy"])
        self.assertIsNone(result["evidence"]["column"]["accuracy"])

    def test_empty_denominators_are_not_evaluated(self):
        result = _score_line_document("opaque", "T02", [], [], self.page_sizes, "as400-specialized")
        self.assertIsNone(result["detection"]["precision"])
        self.assertIsNone(result["detection"]["recall"])
        self.assertIsNone(result["classification"]["accuracy"])
        self.assertIsNone(result["fields"]["decimal"]["accuracy"])

    def test_unknown_document_never_calls_a_specialized_line_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "opaque.pdf").write_bytes(b"synthetic")
            inventory = {"version": 1, "documents": [{"case_id": "opaque", "source_name": "opaque.pdf"}]}
            line_gold = {"version": 1, "documents": {"opaque": [self.gold(1, 100)]}}
            template_gold = {"version": 1, "cases": [{"case_id": "opaque", "template": "unknown"}]}
            inventory_path, gold_path, template_path = (root / "inventory.json", root / "lines.json",
                                                        root / "templates.json")
            inventory_path.write_text(json.dumps(inventory))
            gold_path.write_text(json.dumps(line_gold))
            template_path.write_text(json.dumps(template_gold))
            page = PDFPage(1, 595, 842, "")
            with patch("scripts.template_evaluation.extract_pdf", return_value=(page,)), \
                 patch("scripts.template_evaluation.extract_pdf_tokens") as tokens, \
                 patch("scripts.template_evaluation.detect_template", return_value=TemplateDetection(
                     "unknown", "router", "1", {}, "insufficient-structural-signals")), \
                 patch("scripts.template_evaluation.template_line_extractor_for") as specialized, \
                 patch("scripts.template_evaluation.extract_invoice", return_value=((), (), ())):
                report = evaluate_template_lines(gold_path, root, inventory_path, template_path)
            tokens.assert_not_called()
            specialized.assert_not_called()
            self.assertEqual(report["corpus_coverage"]["selector"]["expected_unknown_documents"], 1)
            self.assertEqual(report["corpus_coverage"]["selector"]["unknown_abstained_as_expected"], 1)
            self.assertEqual(report["by_template"]["T01"]["as400-specialized"]["detection"]["gold_expected"], 0)
            as400 = next(d for d in report["document_results"] if d["engine"] == "as400-specialized")
            self.assertEqual(as400["detection"]["gold_expected"], 0)
            self.assertEqual(as400["detection"]["false_negatives"], 0)


if __name__ == "__main__":
    unittest.main()
