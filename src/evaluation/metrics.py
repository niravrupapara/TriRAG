import math
import numpy as np
from src.utils.logging import get_logger

SIMILARITY_THRESHOLD = 0.6

logger = get_logger(__name__)

def _match_matrix(results: list, expetected: list, embedder) -> np.ndarray:
    """Build a (num_results x num_expected) booleab matrix: does chunk i match fact j?"""

    if not results or not expetected:
        return np.zeros((len(results), len(expetected)), dtype=bool)
    
    chunk_texts = [r["chunk"] for r in results]
    chunks_vectors = embedder.embed(chunk_texts)

    fact_vectors = embedder.embed(expetected)
    similarities = np.dot(chunks_vectors, fact_vectors.T)

    matches = np.zeros((len(results), len(expetected)), dtype=bool)
    for i, chunk in enumerate(chunk_texts):
        for j, fact in enumerate(expetected):
            lexical_match = fact.lower() in chunk.lower()
            semantic_match = similarities[i,j] >= SIMILARITY_THRESHOLD
            matches[i,j] = lexical_match or semantic_match

    return matches

def hit_rate(matches: np.ndarray) -> float:
    """Return 1.0 if any retrieved chunk matches any expected fact."""
    if matches.size == 0:
        return 0.0
    
    return 1.0 if matches.any() else 0.0
   

def mrr(matches: np.ndarray) -> float:
    """Return reciprocal rank of the first chunk that matches any expected fact."""
    if matches.size == 0:
        return 0.0
    relevant_per_chunk = matches.any(axis=1)
    for i, is_relevant in enumerate(relevant_per_chunk):
        if is_relevant:
            return 1.0 / (i + 1)
    
    return 0.0

def ndcg(matches: np.ndarray) -> float:
    """Return NDCG score, rewarding relevant chunks ranked higher."""
    if matches.size == 0:
        return 0.0
    relevant_per_chunk = matches.any(axis=1)
    relevant_count = int(relevant_per_chunk.sum())
    
    if relevant_count == 0:
        return 0.0
    
    dcg = sum(
        1.0 / math.log2(i+2)
        for i, is_relevant in enumerate(relevant_per_chunk) if is_relevant
    )

    idcg = sum(1.0 / math.log2(rank +1) for rank in range(1, relevant_count +1))

    return dcg / idcg

def precision_at_k(matches: np.ndarray) -> float:
    """Return the fraction of retrieved chunks that matched an expected fact."""
    if matches.shape[0] == 0:
        return 0.0
    relevant_per_chunk = matches.any(axis=1)
    return float(relevant_per_chunk.sum()) / matches.shape[0]


def recall(matches: np.ndarray) -> float:
    """Return the fraction of expected facts that were matched by any retrieved chunk."""
    if matches.shape[1] == 0:
        return 0.0
    fact_covered = matches.any(axis=0)
    return float(fact_covered.sum()) / matches.shape[1]

def compute_metrics(results: list, expected: list, embedder) -> dict:
    """Compute hit rate, MRR, NDCG, precision, and recall for a single query."""
    logger.debug(f"Computing metrics for expected: '{expected}' | results: {len(results)}")
    
    matches = _match_matrix(results, expected, embedder)
    metrics = {
        "hit_rate": hit_rate(matches),
        "mrr": mrr(matches),
        "ndcg": ndcg(matches),
        "precision": precision_at_k(matches),
        "recall": recall(matches)
    }

    logger.debug(f"Metrics | {metrics}")
    return metrics
