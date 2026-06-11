from typing import List
from sentence_transformers import CrossEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)

class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model."""

    def __init__(self, model_name: str):
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info(f"Cross-encoder model loaded: {model_name}")

    def rerank(self, query: str, chunks: List[dict], top_k: int) -> List[dict]:
        """Rerank chunks by scoring (query, chunks) pairs with cross-encoder."""
        logger.info(f"Reranking {len(chunks)} chunks with cross-encoder....")

        pairs = [[query, r["chunk"]] for r in chunks]
        scores = self.model.predict(pairs)

        for i, r in enumerate(chunks):
            r["rerank_score"] = float(scores[i])

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        logger.info(f"Reranking done | top score: {reranked[0]['rerank_score']:.4f}")
        return reranked
    
