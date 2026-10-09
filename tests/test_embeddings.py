import unittest
from unittest.mock import patch

from embeddings import DIMENSIONS, QUERY_INSTRUCTION, available_models, embed, embed_file, embed_many


class Response:
    def __init__(self, vector):
        self.vector = vector

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"embedding": self.vector}]}


class ModelsResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": "z-model"}, {"id": "a-model"}, {"name": "ignored"}]}


class BatchResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"index": 1, "embedding": [1.0] * DIMENSIONS},
                         {"index": 0, "embedding": [0.0] * DIMENSIONS}]}


class EmbeddingTests(unittest.TestCase):
    @patch("embeddings.httpx.get", return_value=ModelsResponse())
    def test_discovers_and_sorts_model_ids(self, get):
        self.assertEqual(available_models(), ["a-model", "z-model"])

    @patch("embeddings.settings", return_value={"provider": "openai-compatible", "base_url": "http://embed/v1",
                                                 "model": "test-model", "api_key": ""})
    @patch("embeddings.httpx.post")
    def test_query_adds_retrieval_instruction_and_checks_vector(self, post, _settings):
        post.return_value = Response([0.0] * DIMENSIONS)

        vector = embed("reparación de impresoras", query=True)

        self.assertEqual(len(vector), DIMENSIONS)
        self.assertEqual(post.call_args.kwargs["json"]["input"], f"{QUERY_INSTRUCTION} reparación de impresoras")

    @patch("embeddings.settings", return_value={"provider": "openai-compatible", "base_url": "http://embed/v1",
                                                  "model": "test-model", "api_key": ""})
    @patch("embeddings.httpx.post", return_value=BatchResponse())
    def test_embed_many_batches_text_and_restores_provider_order(self, post, _settings):
        vectors = embed_many(["primer pasaje", "segundo pasaje"])

        self.assertEqual(vectors[0][0], 0.0)
        self.assertEqual(vectors[1][0], 1.0)
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["primer pasaje", "segundo pasaje"])

    @patch("embeddings.settings", return_value={"provider": "jina-local", "base_url": "http://jina/v1",
                                                  "model": "jinaai/jina-embeddings-v5-text-small",
                                                  "api_key": "test-key"})
    @patch("embeddings.httpx.post")
    def test_jina_selects_query_and_document_input_types(self, post, _settings):
        post.return_value = Response([0.0] * DIMENSIONS)

        embed("mantenimiento", query=True)
        self.assertEqual(post.call_args.kwargs["json"]["input_type"], "query")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

        embed("Texto extraído del PDF")
        self.assertEqual(post.call_args.kwargs["json"]["input_type"], "document")
        self.assertEqual(post.call_args.kwargs["json"]["input"], "Texto extraído del PDF")

    @patch("embeddings.settings", return_value={"provider": "jina-local", "document_mode": "direct",
                                                  "base_url": "http://jina/v1", "model": "jina-omni",
                                                  "api_key": "test-key"})
    @patch("embeddings.httpx.post")
    def test_jina_direct_encodes_pdf_and_image_payloads(self, post, _settings):
        import base64

        post.return_value = Response([0.0] * DIMENSIONS)
        embed_file(b"%PDF-test", "application/pdf", "invoice-1")
        pdf = post.call_args.kwargs["json"]["input"][0]
        self.assertEqual(pdf, {"type": "pdf", "document_id": "invoice-1",
                               "data": base64.b64encode(b"%PDF-test").decode("ascii")})

        embed_file(b"image-bytes", "image/png", "image-1")
        image = post.call_args.kwargs["json"]["input"][0]
        self.assertEqual(image, {"type": "image", "data": base64.b64encode(b"image-bytes").decode("ascii")})

    @patch("embeddings.httpx.post")
    def test_rejects_wrong_dimensions(self, post):
        post.return_value = Response([0.0])
        with self.assertRaises(ValueError):
            embed("Document: prueba")


if __name__ == "__main__":
    unittest.main()
