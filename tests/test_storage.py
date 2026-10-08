import unittest
from unittest.mock import Mock, patch

import storage


class StorageTests(unittest.TestCase):
    def test_creates_separate_passage_index_with_compatible_vectors(self):
        es = Mock()
        es.indices.exists.return_value = False

        storage.initialize_index(es)

        self.assertEqual(es.indices.create.call_count, 2)
        passage_index = es.indices.create.call_args.kwargs["index"]
        mapping = es.indices.create.call_args.kwargs["mappings"]["properties"]
        self.assertEqual(passage_index, storage.PASSAGES_INDEX)
        self.assertEqual(mapping["embedding"]["dims"], storage.EMBEDDING_DIMENSIONS)
        self.assertEqual(mapping["start_offset"]["type"], "integer")

    @patch.object(storage, "INDEX", "invoices-new")
    @patch.object(storage, "EMBEDDING_DIMENSIONS", 768)
    def test_rejects_existing_index_with_different_vector_dimensions(self):
        es = Mock()
        es.indices.exists.return_value = True
        es.indices.get_mapping.return_value = {
            "invoices-new": {"mappings": {"properties": {"embedding": {"dims": 1024}}}}
        }

        with self.assertRaisesRegex(ValueError, "1024 dimensiones.*768"):
            storage.initialize_index(es)

        es.indices.put_mapping.assert_not_called()


if __name__ == "__main__":
    unittest.main()
