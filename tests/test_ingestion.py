import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

from src.ingestion.pipeline import IngestionPipeline


class IngestionPipelineTests(unittest.TestCase):
    def test_ingest_builds_and_persists_all_indexes(self):
        config = {
            "chunking": {
                "strategy": "fixed",
                "chunk_size": 5,
                "chunk_overlap": 0,
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
            },
        }
        embedder = Mock()
        embedder.embed.return_value = np.array([[1.0], [2.0]])
        vector_store = Mock()
        bm25_store = Mock()
        bm25_index = object()
        bm25_store.save.return_value = bm25_index
        graph_store = Mock()
        graph = nx.DiGraph()

        pipeline = IngestionPipeline(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
            bm25_store=bm25_store,
            graph_store=graph_store,
        )

        with (
            patch("src.ingestion.pipeline.load_document", return_value="abcde12345"),
            patch("src.ingestion.pipeline.extract_all_graph_documents", return_value=["graph-doc"]),
            patch("src.ingestion.pipeline.build_graph", return_value=graph),
        ):
            result = pipeline.ingest("document.txt")

        self.assertEqual(result.chunks, ["abcde", "12345"])
        self.assertIs(result.bm25_index, bm25_index)
        self.assertIs(result.graph, graph)
        vector_store.save.assert_called_once()
        bm25_store.save.assert_called_once_with(["abcde", "12345"], k1=1.5, b=0.75)
        graph_store.save.assert_called_once_with(graph)


if __name__ == "__main__":
    unittest.main()
