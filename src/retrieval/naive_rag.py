import numpy as np
from typing import List
from src.retrieval.base import BaseRetriever
from src.storage.vector_store import VectorStore       
from src.embeddings.embedder import Embedder
from src.utils.logging import get_logger

logger = get_logger(__name__)

class NaiveRAG(BaseRetriever):
    """Retriever using cosine similarity on dense vector embeddings."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """Initialize with a vector store and an embedder."""
        self.embedder = embedder
        self.vector_store = vector_store
        logger.debug("NaiveRAG initialized with provided vector store and embedder.")

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrieve top-k chunks most similar to the query."""
        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:60]}...")

        docs_with_scores = self.vector_store.index.similarity_search_with_score(query, k = top_k)

        results = []

        for doc, score in docs_with_scores:
            results.append({
                "chunk": doc.page_content,
                "score": float(score)
            }
            )

        logger.info(f"Retrieved {len(results)} chunks | top score: {results[0]['score']:.4f}")
        return results
    
