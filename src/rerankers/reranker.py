from sentence_transformers import CrossEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Reranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder reranker: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=3):
        logger.info(f"Reranking {len(chunks)} candidate chunks to top-{top_k}")
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk for chunk, _ in ranked[:top_k]]