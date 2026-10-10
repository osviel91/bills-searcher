import unittest
from decimal import Decimal

from domain.structured_extraction import BoundingBox, Evidence, ExtractedField, PDFPage
from scripts.template_evaluation import _aggregate, _compare_fields, _normalize


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


if __name__ == "__main__":
    unittest.main()
