import unittest
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from scripts.extraction_evaluation import compare_case, normalize


def field(name, value=None, evidence=()):
    return SimpleNamespace(name=name, value=value, evidence=evidence)


def ev(quote="Total: 12,10", page=1):
    return SimpleNamespace(quote=quote, page_number=page)


class ExtractionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "synthetic-1", "expected_fields": {"total": "12.10", "issue_date": "03/09/2026"},
            "expected_lines": [{"description": "Plan A", "values": {"amount": "12.10"},
                                "evidence": [{"page": 1, "quote": "Plan A row"}]},
                               {"description": "Plan B", "values": {"amount": "2.00"},
                                "evidence": [{"page": 1, "quote": "Plan B row"}]}],
            "evidence": {"total": [{"page": 1, "quote": "Total: 12,10"}],
                         "issue_date": [{"page": 1, "quote": "03/09/2026"}]},
        }
        self.fields = [field("total", Decimal("12.10"), [ev()]),
                       field("issue_date", date(2026, 9, 3), [ev("03/09/2026")])]
        self.lines = [dict(line, evidence=[ev(line["evidence"][0]["quote"])]) for line in self.case["expected_lines"]]

    def test_exact_match_and_date_money_normalization(self):
        result = compare_case(self.case, self.fields, self.lines)
        self.assertEqual(result["field_accuracy"], 1)
        self.assertEqual(result["line_recall"], 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(normalize("total", "12,10"), normalize("total", "12.10"))
        self.assertNotEqual(normalize("total", "12.10"), normalize("total", "12.11"))

    def test_missing_and_wrong_fields_are_distinct(self):
        missing = compare_case(self.case, [field("issue_date", self.fields[1].value, self.fields[1].evidence)], self.lines)
        wrong = compare_case(self.case, [field("total", Decimal("12.11"), [ev()]), self.fields[1]], self.lines)
        self.assertIn("omitted_field", [item["type"] for item in missing["errors"]])
        self.assertIn("wrong_field_value", [item["type"] for item in wrong["errors"]])

    def test_values_assigned_to_the_wrong_field_are_reported_separately(self):
        self.case["expected_fields"]["vat"] = "2.10"
        result = compare_case(self.case, [field("total", Decimal("2.10"), [ev()]),
                                          field("vat", Decimal("12.10"), [ev()]),
                                          self.fields[1]], self.lines)
        self.assertEqual(sum(item["type"] == "misassociated_field" for item in result["errors"]), 2)

    def test_invented_field_and_wrong_evidence_are_reported(self):
        self.case["expected_fields"]["customer_id"] = None
        self.case["evidence"]["customer_id"] = []
        result = compare_case(self.case, self.fields + [field("customer_id", "C-9")], self.lines)
        self.assertIn("invented_field", [item["type"] for item in result["errors"]])
        wrong = compare_case(self.case, [field("total", Decimal("12.10"), [ev("Invoice date: 03/09/2026")]), self.fields[1]], self.lines)
        self.assertIn("incorrect_evidence", [item["type"] for item in wrong["errors"]])

    def test_insufficient_reference_evidence_is_visible(self):
        del self.case["evidence"]["issue_date"]
        result = compare_case(self.case, self.fields, self.lines)
        self.assertIn("issue_date", result["insufficient_evidence"])

    def test_duplicate_and_reordered_lines(self):
        duplicate = compare_case(self.case, self.fields, self.lines + [self.lines[0]])
        reordered = compare_case(self.case, self.fields, list(reversed(self.lines)))
        self.assertIn("duplicate_or_incorrect_line", [item["type"] for item in duplicate["errors"]])
        self.assertEqual(reordered["line_precision"], 1)
        self.assertEqual(reordered["line_recall"], 1)

    def test_one_cent_line_difference_is_not_tolerated(self):
        changed = [dict(self.lines[0], values={"amount": "12.11"}), self.lines[1]]
        result = compare_case(self.case, self.fields, changed)
        self.assertEqual(result["line_recall"], 0.5)
        self.assertIn("incorrect_line_attributes", [item["type"] for item in result["errors"]])

    def test_line_attributes_assigned_to_another_description_are_reported(self):
        swapped = [{"description": "Plan A", "values": {"amount": "2.00"}},
                   {"description": "Plan B", "values": {"amount": "12.10"}}]
        result = compare_case(self.case, self.fields, swapped)
        self.assertIn("misassociated_line", [item["type"] for item in result["errors"]])


if __name__ == "__main__":
    unittest.main()
