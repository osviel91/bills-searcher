import unittest
from unittest.mock import patch

from app import normalize_pdf, reciprocal_rank_fusion
import storage


class PdfValidationTests(unittest.TestCase):
    def test_strips_short_multipart_preamble(self):
        wrapped = b"--boundary\r\nContent-Type: multipart/form-data\r\n\r\n%PDF-1.4\r\nbody"
        self.assertEqual(normalize_pdf(wrapped), b"%PDF-1.4\r\nbody")

    def test_rejects_missing_or_late_pdf_signature(self):
        with self.assertRaises(ValueError):
            normalize_pdf(b"not a PDF")
        with self.assertRaises(ValueError):
            normalize_pdf(b" " * 1024 + b"%PDF-1.4")


class HybridRankingTests(unittest.TestCase):
    def test_document_ranked_by_both_retrievers_wins(self):
        def hit(doc_id):
            return {"_id": doc_id, "_source": {"filename": f"{doc_id}.pdf"}, "_score": 1.0}

        results = reciprocal_rank_fusion([hit("exact"), hit("phrase")],
                                         [hit("semantic"), hit("exact")], limit=3)

        self.assertEqual(results[0]["_source"]["filename"], "exact.pdf")


class ElasticsearchConfigurationTests(unittest.TestCase):
    @patch("storage.Elasticsearch")
    def test_prefers_api_key_when_configured(self, constructor):
        with patch.object(storage, "ES_API_KEY", "secret"), patch.object(storage, "ES_USERNAME", "user"), patch.object(storage, "ES_PASSWORD", "password"):
            storage.client()
        self.assertEqual(constructor.call_args.kwargs["api_key"], "secret")
        self.assertNotIn("basic_auth", constructor.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
