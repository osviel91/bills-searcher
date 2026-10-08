import unittest

from passages import chunk_text


class PassageTests(unittest.TestCase):
    def test_chunks_keep_exact_offsets_and_respect_size(self):
        text = "first line\nsecond line\n" + ("x" * 25)
        chunks = chunk_text(text, max_chars=14)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 14)
            self.assertEqual(text[chunk["start_offset"]:chunk["end_offset"]], chunk["text"])
        self.assertEqual("".join(chunk["text"] for chunk in chunks).replace("\n", ""),
                         text.replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
