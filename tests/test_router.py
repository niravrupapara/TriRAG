import unittest

from src.routing.router import classify_query


class RouterTests(unittest.TestCase):
    def test_relationship_question_routes_to_graph(self):
        self.assertEqual(
            classify_query("How does backpropagation relate to gradient descent?"),
            "graph",
        )

    def test_short_query_routes_to_hybrid(self):
        self.assertEqual(classify_query("gradient descent"), "hybrid")

    def test_default_query_routes_to_naive(self):
        self.assertEqual(
            classify_query("Explain the training process for a neural network"),
            "naive",
        )


if __name__ == "__main__":
    unittest.main()
