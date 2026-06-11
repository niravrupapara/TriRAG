import yaml
from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.storage.vector_store import VectorStore
from src.retrieval.naive_rag import NaiveRAG

from src.storage.bm25_store import BM25Store
from src.retrieval.bm25_retriever import BM25RAG
from src.retrieval.hybrid_rag import HybridRAG
from src.rerankers.cross_encoder import CrossEncoderReranker

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
    
    embedder = Embedder(config['embeddings']['model'])
    vectors = embedder.embed(chunks)

    store = VectorStore(config["paths"]["vector"])
    store.save(vectors, chunks)

    logger.info(f"Ingestion completed | chunks saved: {len(chunks)}")
    return store, embedder

def query(question: str, store: VectorStore, embedder: Embedder, config: dict):
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

    embedder = Embedder(config['embeddings']['model'])
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


def hybrid_query(question: str, store: VectorStore, embedder: Embedder,
                 bm25_index, bm25_chunks: list, config: dict):
    """Retrieve using Hybrid RAG + cross-encoder reranking."""
    logger.info(f"Hybrid query received: {question}")

    dense_retriever = NaiveRAG(store, embedder)
    bm25_retriever = BM25RAG(bm25_index, bm25_chunks)
    hybrid_retriever = HybridRAG(bm25_retriever, dense_retriever)

    candidates = hybrid_retriever.retrieve(question, top_k=config["retrieval"]["final_top_k"] * 2)

    reranker = CrossEncoderReranker(config["reranker"]["model"])
    results = reranker.rerank(question, candidates, top_k=config["reranker"]["top_k"])

    print(f"\n{'='*60}")
    print(f"Hybrid Query: {question}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} | RRF Score: {r['score']:.6f} | Rerank Score: {r['rerank_score']:.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")



if __name__ == "__main__":
    config = load_config("config.yaml")
    store, embedder, bm25_index, bm25_chunks = hybrid_ingest(r"C:\Users\Nirav Rupapara\Downloads\test_fnn_pyq.pdf", config)
    hybrid_query("PCA principal component analysis", store, embedder, bm25_index, bm25_chunks, config)

