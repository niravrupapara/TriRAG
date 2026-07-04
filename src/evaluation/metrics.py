import math
from src.utils.logging import get_logger

logger = get_logger(__name__)

def hit_rate(results: list, expected: str) -> float:
    """Return 1.0 if expected keyword found in any result chunk, else 0.0."""
    for r in results:
        if expected.lower() in r["chunk"].lower():
            logger.debug(f"Hit found | expected: '{expected}'")
            return 1.0
    logger.debug(f"No hit | expected: '{expected}'")
    return 0.0

def mrr(results: list, expected: str) -> float:
    """Return reciprocal rank of first result containing expected keyword."""
    for i, r in enumerate(results):
        if expected.lower() in r["chunk"].lower():
            rank = i+1
            logger.debug(f"MRR hit at rank {rank} | expected: '{expected}'")
            return 1.0 / rank
    logger.debug(f"Expected chunk not found in results: {expected}")
    return 0.0

def ndcg(results: list, expected: str) -> float:
    """Return NDCG score for results given expected keyword."""
    for i, r in enumerate(results):
        if expected.lower() in r["chunk"].lower():
            rank = i + 1
            dcg = 1.0 / math.log2(rank + 1)
            idcg = 1.0 / math.log2(2)
            ndcg_score = dcg / idcg
            logger.debug(f"NDCG hit at rank {rank} | expected: '{expected}'")
            return ndcg_score   
    logger.debug(f"Expected chunk not found in results: {expected}")
    return 0.0

def compute_metrics(results: list, expected: str) -> dict:
    """Compute hit rate, MRR, and NDCG for a single query."""
    logger.debug(f"Computing metrics for expected: '{expected}' | results: {len(results)}")
    scores = {
        "hit_rate": hit_rate(results, expected),
        "mrr": mrr(results, expected),
        "ndcg": ndcg(results, expected)
    }

    logger.debug(f"Metrics | hit_rate: {scores['hit_rate']}, mrr: {scores['mrr']}, ndcg: {scores['ndcg']}")
    return scores