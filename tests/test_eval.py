import unittest

import numpy as np

from src.evaluation.metrics import compute_metrics, hit_rate, mrr, ndcg, precision_at_k, recall


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            if "gradient descent" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors)


class MetricTests(unittest.TestCase):
    def test_metrics_handle_empty_matches(self):
        matches = np.zeros((0, 0), dtype=bool)

        self.assertEqual(hit_rate(matches), 0.0)
        self.assertEqual(mrr(matches), 0.0)
        self.assertEqual(ndcg(matches), 0.0)
        self.assertEqual(precision_at_k(matches), 0.0)
        self.assertEqual(recall(matches), 0.0)

    def test_compute_metrics_detects_lexical_match(self):
        results = [{"chunk": "The optimizer uses gradient descent.", "score": 1.0}]
        expected = ["gradient descent"]

        metrics = compute_metrics(results, expected, FakeEmbedder())

        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
