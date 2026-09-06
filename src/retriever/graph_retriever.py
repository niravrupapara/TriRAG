from src.graph.traversal import query_graph
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRetriever:

    def __init__(self, graph, chunks, embedder):
        self.graph = graph
        self.chunks = chunks
        self.embedder = embedder

    def retrieve(self, query, k=5):
        logger.info(f"Graph traversal retrieval for query: '{query}' (k={k})")
        return query_graph(
            query,
            self.graph,
            self.chunks,
            self.embedder,
            top_k=k
        )