from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.bm25_retriever import BM25RAG
from src.retrieval.hybrid_rag import HybridRAG
from src.rerankers.remote_reranker import RemoteReranker
from src.retrieval.graph_rag import GraphRAG

from src.utils.logging import get_logger

logger = get_logger(__name__)

GRAPH_KEYWORDS = [
    "relates", "connects", "between", "how does", "relationship",
    "connection", "linked", "associated", "depends", "leads to"
]

def classify_query(question: str) -> str:
    """Return 'graph', 'hybrid', or 'naive' based on question content."""
    question_lower = question.lower()

    for keyword in GRAPH_KEYWORDS:
        if keyword in question_lower:
            logger.debug(f"Query classified as: graph | matched keyword: '{keyword}'")
            return "graph"
        
    if len(question_lower.split()) <= 3:
        logger.debug("Query classified as: hybrid | reason: short query")
        return "hybrid"
    
    logger.debug("Query classified as: naive | reason: default")
    return "naive"



def route_query(question: str, store, embedder, bm25_index, bm25_chunks: list, graph, config: dict) -> tuple:
    """Route question to best strategy. Returns (strategy_name, results)."""
    strategy = classify_query(question)
    logger.info(f"Routing query | strategy: {strategy} | question: {question[:60]}")

    if strategy == "graph":
        retriever = GraphRAG(graph, config, embedder)
        results = retriever.retrieve(question, top_k=config["graph"]["top_k"])

    elif strategy == "hybrid":
        dense_retriever = NaiveRAG(store, embedder)
        bm25_retriever = BM25RAG(bm25_index, bm25_chunks)
        hybrid_retriever = HybridRAG(bm25_retriever, dense_retriever)
        candidates = hybrid_retriever.retrieve(
            question, top_k=config["retrieval"]["final_top_k"] *2
        )
        reranker = RemoteReranker()
        results = reranker.rerank(
            question, candidates, top_k=config["reranker"]["top_k"]
        )

    else:
        retriever = NaiveRAG(store, embedder)
        results = retriever.retrieve(
            question, top_k=config["retrieval"]["dense_top_k"]
        )

    logger.info(f"Routing complete | strategy: {strategy} | results: {len(results)}")
    return strategy, results