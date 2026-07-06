import json
import time

from src.evaluation.metrics import compute_metrics
from src.evaluation.llm_judge import judge
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.bm25_retriever import BM25RAG
from src.retrieval.hybrid_rag import HybridRAG
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.retrieval.graph_rag import GraphRAG
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_questions(path: str) -> list:
    """Load eval questions from JSON file."""
    with open(path, "r") as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} eval questions from: {path}")
    return questions


def evaluate(store, embedder, bm25_index, bm25_chunks: list,
             graph, config: dict) -> dict:
    """Run all 3 strategies on all questions. Returns averaged metrics per strategy."""
    questions_path = config["paths"]["eval"] + "/questions.json"
    questions = load_questions(questions_path)

    strategies = ["naive", "hybrid", "graph"]
    scores = {s: {"hit_rate": [], "mrr": [], "ndcg": [], "llm_score": []} for s in strategies}

    reranker = CrossEncoderReranker(config["reranker"]["model"])
    logger.info("CrossEncoderReranker loaded.")

    for item in questions:
        question = item["question"]
        expected = item["expected"]
        logger.info(f"Evaluating | question: {question[:60]}")

        naive_results = NaiveRAG(store, embedder).retrieve(
            question, top_k=config["retrieval"]["dense_top_k"]
        )
        for k, v in compute_metrics(naive_results, expected).items():
            scores["naive"][k].append(v)
        scores["naive"]["llm_score"].append(judge(question, naive_results, config))

        dense = NaiveRAG(store, embedder)
        bm25 = BM25RAG(bm25_index, bm25_chunks)
        candidates = HybridRAG(bm25, dense).retrieve(
            question, top_k=config["retrieval"]["final_top_k"] * 2
        )
        hybrid_results = reranker.rerank(
            question, candidates, top_k=config["reranker"]["top_k"]
        )
        for k, v in compute_metrics(hybrid_results, expected).items():
            scores["hybrid"][k].append(v)
        scores["hybrid"]["llm_score"].append(judge(question, hybrid_results, config))

        graph_results = GraphRAG(graph, config).retrieve(
            question, top_k=config["graph"]["top_k"]
        )
        for k, v in compute_metrics(graph_results, expected).items():
            scores["graph"][k].append(v)
        scores["graph"]["llm_score"].append(judge(question, graph_results, config))
        time.sleep(3)

    summary = {}
    for strategy in strategies:
        summary[strategy] = {
            metric: round(sum(vals) / len(vals), 4) if vals else 0.0
            for metric, vals in scores[strategy].items()
        }
        logger.info(f"Strategy: {strategy} | avg scores: {summary[strategy]}")

    return summary


def print_report(summary: dict) -> None:
    """Print comparison table of all strategies and metrics."""
    print(f"\n{'='*65}")
    print(f"{'EVALUATION REPORT':^65}")
    print(f"{'='*65}")
    print(f"{'Strategy':<12} {'Hit Rate':>10} {'MRR':>10} {'NDCG':>10} {'LLM Score':>12}")
    print(f"{'-'*65}")
    for strategy, metrics in summary.items():
        print(
            f"{strategy.upper():<12}"
            f"{metrics['hit_rate']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['ndcg']:>10.4f}"
            f"{metrics['llm_score']:>12.4f}"
        )
    print(f"{'='*65}\n")
