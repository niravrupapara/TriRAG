from src.utils.logging import get_logger

logger = get_logger(__name__)


class RRF:

    def __init__(self, k=60):
        self.k = k

    def fuse(self, naive_chunks, bm25_chunks, graph_chunks=None, top_k=10):
        logger.info(f"Fusing candidates via Reciprocal Rank Fusion (k={self.k}, top_k={top_k})")
        scores = {}

        for chunks in [naive_chunks, bm25_chunks, graph_chunks or []]:
            for rank, chunk in enumerate(chunks, start=1):
                scores[chunk] = scores.get(chunk, 0) + 1 / (self.k + rank)

        ranked_chunks = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        return ranked_chunks[:top_k]