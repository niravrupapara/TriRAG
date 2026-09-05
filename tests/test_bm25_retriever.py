import unittest

from rank_bm25 import BM25Okapi

from src.retrieval.bm25_retriever import BM25RAG


class BM25RetrieverTests(unittest.TestCase):
    def test_retrieve_returns_best_matching_chunks_first(self):
        chunks = [
            "apples and bananas",
            "neural networks use gradient descent",
            "database indexes speed up lookup",
        ]
        tokenized = [chunk.lower().split() for chunk in chunks]
        retriever = BM25RAG(BM25Okapi(tokenized), chunks)

        results = retriever.retrieve("gradient descent", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk"], "neural networks use gradient descent")
        self.assertIn("score", results[0])


if __name__ == "__main__":
    unittest.main()
