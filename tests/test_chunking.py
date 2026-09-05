import unittest

from src.chunking.chunker import chunk_text, fixed_chunk, recursive_chunk


class ChunkingTests(unittest.TestCase):
    def test_fixed_chunk_uses_overlap(self):
        chunks = fixed_chunk("abcdefghij", chunk_size=4, overlap=1)

        self.assertEqual(chunks, ["abcd", "defg", "ghij", "j"])

    def test_recursive_chunk_preserves_short_text(self):
        text = "short paragraph"

        self.assertEqual(recursive_chunk(text, chunk_size=50, overlap=5), [text])

    def test_chunk_text_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", strategy="unknown", chunk_size=10, overlap=0)


if __name__ == "__main__":
    unittest.main()
