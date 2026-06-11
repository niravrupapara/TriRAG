from typing import List
from src.retrieval.base import BaseRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

class HybridRAG(BaseRetriever):
    """Retriever combining BM25 and dense retrieval using RRF."""
    def __init__(self, bm25_retriever: BaseRetriever, dense_retriever: BaseRetriever):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        logger.debug("HybridRAG initialized")

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrieve top-k chunks by merging BM25 and dense results with RRF."""
        logger.info(f"HybridRAG retrieving for query: {query[:60]}..")

        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k*2)

        bm25_ranks = {r["chunk"]: rank + 1 for rank, r in enumerate(bm25_results)}
        dense_ranks = {r["chunk"]: rank +1 for rank, r in enumerate(dense_results)}

        all_chunks = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        k=60
        rrf_scores = {}
        for chunk in all_chunks:
            score = 0.0
            if chunk in bm25_ranks:
                score+=1 / (k + bm25_ranks[chunk])
            if chunk in dense_ranks:
                score += 1 / (k + dense_ranks[chunk])

            rrf_scores[chunk] = score

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x:x[1], reverse=True)[:top_k]
        results = [{"chunk": chunk, "score": round(score,6)} for chunk, score in sorted_chunks]

        logger.info(f"HybridRAG merged {len(all_chunks)} unique chunks | returning top {len(results)}")
        return results