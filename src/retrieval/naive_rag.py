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

        query_vector = self.embedder.embed_one(query)
        scores = np.dot(self.vector_store.vectors, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [] 
        for i in top_indices:
            results.append({
                "chunk": self.vector_store.chunks[i],
                "score": float(scores[i])

            })

        logger.info(f"Retrieved {len(results)} chunks | top score: {results[0]['score']:.4f}")
        return results
    
