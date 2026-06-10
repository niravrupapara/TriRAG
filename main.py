import yaml
from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.storage.vector_store import VectorStore
from src.retrieval.naive_rag import NaiveRAG
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
    results = retriever.retrieve(question, top_k=config['retrieval']['top_k'])

    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} | Score: {r['score']:.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")

if __name__ == "__main__":
    config = load_config("config.yaml")
    store, embedder = ingest(r"C:\Users\Nirav Rupapara\Downloads\test_fnn_pyq.pdf", config)
    query("PCa principle component analysis", store, embedder, config)

