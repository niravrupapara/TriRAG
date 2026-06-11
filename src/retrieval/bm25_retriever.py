from typing import List
from rank_bm25 import BM25Okapi
from src.retrieval.base import BaseRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BM25RAG(BaseRetriever):
    """Retriever using BM25 keyword scoring."""

    def __init__(self, index: BM25Okapi, chunks: List[str]):
        self.index = index
        self.chunks = chunks
        logger.debug("BM25RAG initialized.")

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrieve top-k chunks using BM25 keyword scoring."""
        logger.info(f"BM25 retrieving top-{top_k} for query: {query[:60]}....")

        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [{"chunk": self.chunks[i], "score": float(scores[i])} for i in top_indices]

        logger.info(f"BM25 retrieved {len(results)} chunks | top score: {results[0]['score']:.4f}")

        return results
    