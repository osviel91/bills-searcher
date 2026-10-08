import unittest
from unittest.mock import patch

from embeddings import DIMENSIONS, QUERY_INSTRUCTION, available_models, embed


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


class EmbeddingTests(unittest.TestCase):
    @patch("embeddings.httpx.get", return_value=ModelsResponse())
    def test_discovers_and_sorts_model_ids(self, get):
        self.assertEqual(available_models(), ["a-model", "z-model"])

    @patch("embeddings.httpx.post")
    def test_query_adds_retrieval_instruction_and_checks_vector(self, post):
        post.return_value = Response([0.0] * DIMENSIONS)

        vector = embed("reparación de impresoras", query=True)

        self.assertEqual(len(vector), DIMENSIONS)
        self.assertEqual(post.call_args.kwargs["json"]["input"], f"{QUERY_INSTRUCTION} reparación de impresoras")

    @patch("embeddings.httpx.post")
    def test_rejects_wrong_dimensions(self, post):
        post.return_value = Response([0.0])
        with self.assertRaises(ValueError):
            embed("Document: prueba")


if __name__ == "__main__":
    unittest.main()
