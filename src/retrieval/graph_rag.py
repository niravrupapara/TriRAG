import networkx as nx
from typing import List
from src.retrieval.base import BaseRetriever
from src.graph.traversal import query_graph
from src.utils.logging import get_logger

logger  = get_logger(__name__)

class GraphRAG(BaseRetriever):
    """Retriever using knowledge graph traversal."""

    def __init__(self, graph: nx.DiGraph, config: dict):
        self.graph = graph
        self.config = config
        logger.debug("GraphRAG initialized.")

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrive relevant chunks by traversing the knowledge graph."""
        logger.info(f"GraphRAG retrieving for query: {query[:60]}...")

        results = query_graph(query, self.graph, self.config)
        results = results[:top_k]

        logger.info(f"GraphRAG returned {len(results)} results.")
        return results
    