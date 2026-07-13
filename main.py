import os

import yaml
from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_text
from src.embeddings.remote_embedder import RemoteEmbedder

from src.storage.vector_store import VectorStore
from src.retrieval.naive_rag import NaiveRAG

from src.storage.bm25_store import BM25Store
from src.retrieval.bm25_retriever import BM25RAG
from src.retrieval.hybrid_rag import HybridRAG
from src.rerankers.remote_reranker import RemoteReranker

from src.graph.extractor import extract_all_graph_documents
from src.graph.builder import build_graph
from src.storage.graph_store import GraphStore
from src.retrieval.graph_rag import GraphRAG

from src.routing.router import classify_query, route_query

from src.evaluation.evaluator import evaluate, print_report



from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Configuration loaded successfully.")
    return config

def ingest(file_path: str, config: dict):
    """Load, chunk, embed and save a document."""
    logger.info(f"Starting ingestion for file: {file_path}")

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        strategy=config['chunking']['strategy'],
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap']
        )
    
    embedder = RemoteEmbedder()

    vectors = embedder.embed(chunks)

    store = VectorStore(config["paths"]["vector"])
    store.save(vectors, chunks)

    logger.info(f"Ingestion completed | chunks saved: {len(chunks)}")
    return store, embedder

def query(question: str, store: VectorStore, embedder: RemoteEmbedder, config: dict):
    """Retrieve top-k chunks for a question."""
    logger.info(f"Query received: {question}")

    retriever = NaiveRAG(store, embedder)
    results = retriever.retrieve(question, top_k=config['retrieval']['dense_top_k'])

    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} | Score: {r['score']:.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")

def hybrid_ingest(file_path: str, config: dict):
    """Load, chunk, embed, save vectors and build BM25 index."""
    logger.info(f"Starting hybrid ingestion for file: {file_path}")

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        strategy=config['chunking']['strategy'],
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap']
    )

    embedder = RemoteEmbedder()
    vectors = embedder.embed(chunks)

    store = VectorStore(config["paths"]["vector"])
    store.save(vectors, chunks)

    bm25_store = BM25Store(config["paths"]["bm25"])
    bm25_index, bm25_chunks = bm25_store.save(
        chunks,
        k1=config["bm25"]["k1"],
        b=config["bm25"]["b"]
    ), chunks

    logger.info(f"Hybrid ingestion completed | chunks saved: {len(chunks)}")
    return store, embedder, bm25_index, bm25_chunks


def hybrid_query(question: str, store: VectorStore, embedder: RemoteEmbedder,
                 bm25_index, bm25_chunks: list, config: dict):
    """Retrieve using Hybrid RAG + cross-encoder reranking."""
    logger.info(f"Hybrid query received: {question}")

    dense_retriever = NaiveRAG(store, embedder)
    bm25_retriever = BM25RAG(bm25_index, bm25_chunks)
    hybrid_retriever = HybridRAG(bm25_retriever, dense_retriever)

    candidates = hybrid_retriever.retrieve(question, top_k=config["retrieval"]["final_top_k"] * 2)

    reranker = RemoteReranker()
    results = reranker.rerank(question, candidates, top_k=config["reranker"]["top_k"])

    print(f"\n{'='*60}")
    print(f"Hybrid Query: {question}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} | RRF Score: {r['score']:.6f} | Rerank Score: {r['rerank_score']:.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")


def graph_ingest(file_path: str, config: dict):
    """Load, chunk, extract triples, build and save knowledge graph."""
    logger.info(f"Starting graph ingestion for file: {file_path}")

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        strategy=config['chunking']['strategy'],
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap']
    )

    graph_documents = extract_all_graph_documents(chunks, config)
    graph = build_graph(graph_documents)

    graph_store = GraphStore(config["paths"]["graph"])
    graph_store.save(graph)

    logger.info(f"Graph ingestion completed | graph_documents: {len(graph_documents)}")
    return graph


def graph_query(question: str, graph, config: dict):
    """Retrieve using GraphRAG knowledge graph traversal."""
    logger.info(f"Graph query received: {question}")

    retriever = GraphRAG(graph, config)
    results = retriever.retrieve(question, top_k=config["graph"]["top_k"])

    print(f"\n{'='*60}")
    print(f"Graph Query: {question}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")


def full_ingest(file_path: str, config: dict):
    """Load, chunk, embed, build BM25 and knowledge graph for all strategies."""
    logger.info(f"Starting full ingestion for file: {file_path}")

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        strategy=config["chunking"]["strategy"],
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap']
    )

    embedder = RemoteEmbedder()
    vectors = embedder.embed(chunks)

    store = VectorStore(config['paths']['vector'])
    store.save(vectors, chunks)

    bm25_store = BM25Store(config['paths']['bm25'])
    bm25_index = bm25_store.save(
        chunks,
        k1=config['bm25']['k1'],
        b=config['bm25']['b']
    )
    bm25_chunks = chunks

    graph_documents = extract_all_graph_documents(chunks, config)
    graph = build_graph(graph_documents)
    graph_store = GraphStore(config['paths']['graph'])
    graph_store.save(graph)

    logger.info(f"Full ingestion complete | chunks: {len(chunks)} | graph_documents: {len(graph_documents)}")
    return store, embedder, bm25_index, bm25_chunks, graph


def load_or_ingest(file_path: str, config: dict):
    """Load indexes from disk if they exists, else run full ingestion."""

    store = VectorStore(config['paths']['vector'])
    bm25_store = BM25Store(config['paths']['bm25'])
    graph_store = GraphStore(config['paths']['graph'])

    indexes_exist = (
        not store.is_empty() and
        os.path.exists(bm25_store.index_path) and
        os.path.exists(graph_store.graph_path)
    )

    if indexes_exist:
        logger.info("Indexes found on disk. Loading from cache...")
        store.load()
        logger.info("Loading embedding model...")
        embedder = RemoteEmbedder()
        bm25_index, bm25_chunks = bm25_store.load()
        graph = graph_store.load()
        logger.info("Indexes loaded successfully.")

        return store, embedder, bm25_index, bm25_chunks, graph

    logger.info("Indexes not found. Running full ingestion...")
    return full_ingest(file_path, config)


def router_query(question: str, store, embedder, bm25_index, bm25_chunks: list, graph, config: dict):
    """Route question to best RAG strategy and print results."""
    logger.info(f"Router query received: {question}")

    strategy, results = route_query(
        question, store, embedder, bm25_index, bm25_chunks, graph, config
    )

    print(f"\n{'='*60}")
    print(f"Query   : {question}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")


def run_evaluation(store, embedder, bm25_index, bm25_chunks: list, graph, config: dict):
    """Run full evaluation benchmark across all RAG strategies."""
    logger.info("Starting evaluation benchmark...")

    summary = evaluate(
        store, embedder, bm25_index, bm25_chunks, graph, config
    )

    print_report(summary)

    logger.info("Evaluation benchmark complete.")


if __name__ == "__main__":
    config = load_config("config.yaml")
    
    store, embedder, bm25_index, bm25_chunks, graph = load_or_ingest(
        "data/raw/ml_training_pipeline.txt", config
    )
    router_query("What is backpropagation?", store, embedder, bm25_index, bm25_chunks, graph, config)
    # run_evaluation(store, embedder, bm25_index, bm25_chunks, graph, config)

