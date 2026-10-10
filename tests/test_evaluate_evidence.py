import unittest

from evidence_evaluation import invoice_recall_at_k, recall_at_k, scoped_recall_at_k, valid_cases


class EvidenceEvaluationTests(unittest.TestCase):
    def test_recall_counts_labelled_quotes_in_the_correct_invoice(self):
        relevant = [
            {"invoice_id": "invoice-1", "quote": "maintenance of printers"},
            {"invoice_id": "invoice-2", "quote": "web hosting service"},
        ]
        results = [
            {"invoice_id": "invoice-1", "content": "Maintenance\n of printers and equipment"},
            {"invoice_id": "invoice-2", "content": "other text"},
            *({"invoice_id": "other", "content": "irrelevant"} for _ in range(4)),
            {"invoice_id": "invoice-2", "content": "Web hosting service for one year"},
        ]

        self.assertEqual(recall_at_k(relevant, results, 5), 0.5)
        self.assertEqual(recall_at_k(relevant, results, 10), 1.0)

    def test_rejects_cases_without_stable_invoice_and_quote_labels(self):
        self.assertTrue(valid_cases([{"query": "hosting", "relevant_evidence": [
            {"invoice_id": "invoice-1", "quote": "web hosting"},
        ]}]))
        self.assertFalse(valid_cases([{"query": "hosting", "relevant_evidence": [
            {"passage_id": "unstable", "quote": "web hosting"},
        ]}]))

    def test_separates_invoice_resolution_from_scoped_passage_recall(self):
        labels = [
            {"invoice_id": "invoice-1", "quote": "paid by transfer"},
            {"invoice_id": "invoice-2", "quote": "due in 30 days"},
        ]
        self.assertEqual(invoice_recall_at_k(["invoice-1", "invoice-2"], [
            {"invoice_id": "invoice-1"}, {"invoice_id": "other"},
        ], 2), 0.5)
        self.assertEqual(scoped_recall_at_k(labels, {
            "invoice-1": [{"content": "Paid by transfer at issue"}],
            "invoice-2": [{"content": "Due in 30 days"}],
        }, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
