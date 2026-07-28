import os
import gc
import json
import shutil
import uuid
import hashlib
import yaml
from typing import Dict, List, Tuple, Optional, Any

from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_text
from src.embeddings.remote_embedder import RemoteEmbedder
from src.storage.vector_store import VectorStore
from src.storage.bm25_store import BM25Store
from src.storage.graph_store import GraphStore
from src.graph.extractor import extract_all_graph_documents
from src.graph.builder import build_graph
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.bm25_retriever import BM25RAG
from src.retrieval.hybrid_rag import HybridRAG
from src.retrieval.graph_rag import GraphRAG
from src.rerankers.remote_reranker import RemoteReranker
from src.routing.router import route_query
from src.evaluation.evaluator import evaluate, print_report
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config


def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file to detect content changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


class TriRAG:
    """
    Unified, isolated TriRAG Engine orchestrating Ingestion, Storage, Routing,
    Retrieval, Generation, Evaluation, and Memory Management.
    """

    def __init__(self, collection_name: Optional[str] = None, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # Handle anonymous UUID instances vs named collections
        if collection_name is None:
            self.collection_name = f"anon_{uuid.uuid4().hex[:8]}"
            self.is_anonymous = True
        else:
            self.collection_name = collection_name
            self.is_anonymous = False

        self.base_dir = os.path.join("data", "collections", self.collection_name)
        self.vector_path = os.path.join(self.base_dir, "vector")
        self.bm25_path = os.path.join(self.base_dir, "bm25")
        self.graph_path = os.path.join(self.base_dir, "graph")
        self.metadata_path = os.path.join(self.base_dir, "metadata.json")

        self.embedder = RemoteEmbedder()
        self.vector_store = VectorStore(self.vector_path)
        self.bm25_store = BM25Store(self.bm25_path)
        self.graph_store = GraphStore(self.graph_path)

        self.bm25_index = None
        self.bm25_chunks = []
        self.graph = None

        logger.info(
            f"TriRAG Engine initialized | collection: '{self.collection_name}' | "
            f"anonymous: {self.is_anonymous} | path: {self.base_dir}"
        )

    def __enter__(self):
        """Enable Python context manager support (with TriRAG() as rag:)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically unload memory upon exiting with block."""
        self.unload()

    def __del__(self):
        """Auto-delete temp folder for anonymous instances on garbage collection."""
        if getattr(self, "is_anonymous", False) and os.path.exists(getattr(self, "base_dir", "")):
            try:
                shutil.rmtree(self.base_dir)
            except Exception:
                pass

    def full_ingest(self, file_path: str):
        """Load, chunk, embed, and build BM25 + Knowledge Graph for all strategies."""
        logger.info(f"Starting full ingestion for collection '{self.collection_name}' | file: {file_path}")

        text = load_document(file_path)
        chunks = chunk_text(
            text,
            strategy=self.config["chunking"]["strategy"],
            chunk_size=self.config["chunking"]["chunk_size"],
            overlap=self.config["chunking"]["chunk_overlap"]
        )

        vectors = self.embedder.embed(chunks)
        self.vector_store.save(vectors, chunks, self.embedder)

        self.bm25_index = self.bm25_store.save(
            chunks,
            k1=self.config["bm25"]["k1"],
            b=self.config["bm25"]["b"]
        )
        self.bm25_chunks = chunks

        graph_documents = extract_all_graph_documents(chunks, self.config)
        self.graph = build_graph(graph_documents, self.embedder)
        self.graph_store.save(self.graph)

        # Save metadata JSON to validate cache on future runs
        file_hash = calculate_file_hash(file_path)
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "collection_name": self.collection_name,
                "is_anonymous": self.is_anonymous,
                "file_path": file_path,
                "file_hash": file_hash,
                "num_chunks": len(chunks)
            }, f, indent=2)

        logger.info(f"Full ingestion complete | collection: '{self.collection_name}' | chunks: {len(chunks)}")

    def load_or_ingest(self, file_path: str, force_reingest: bool = False):
        """Load indexes from disk if cached & hash matches, else run full ingestion."""
        current_hash = calculate_file_hash(file_path)

        cache_valid = False
        if os.path.exists(self.metadata_path) and not force_reingest:
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("file_hash") == current_hash and meta.get("file_path") == file_path:
                    cache_valid = True
            except Exception as e:
                logger.warning(f"Error reading metadata file: {e}")

        indexes_exist = (
            not self.vector_store.is_empty() and
            os.path.exists(self.bm25_store.index_path) and
            os.path.exists(self.graph_store.graph_path)
        )

        if cache_valid and indexes_exist:
            logger.info(f"Cache hit! Loading indexes for collection '{self.collection_name}'...")
            self.vector_store.load(self.embedder)
            self.bm25_index, self.bm25_chunks = self.bm25_store.load()
            self.graph = self.graph_store.load()
            logger.info(f"Collection '{self.collection_name}' loaded successfully from disk.")
            return

        logger.info(f"Cache miss / file changed! Re-ingesting for collection '{self.collection_name}'...")
        self.full_ingest(file_path)

    def query(self, question: str, strategy: str = "auto") -> Tuple[str, List[dict]]:
        """Query the RAG system using specified strategy or auto-routing."""
        logger.info(f"Querying collection '{self.collection_name}' | strategy: {strategy} | question: {question[:60]}")

        if strategy == "auto":
            return route_query(
                question,
                self.vector_store,
                self.embedder,
                self.bm25_index,
                self.bm25_chunks,
                self.graph,
                self.config
            )
        elif strategy == "graph":
            retriever = GraphRAG(self.graph, self.config, self.embedder)
            results = retriever.retrieve(question, top_k=self.config["graph"]["top_k"])
            return "graph", results
        elif strategy == "hybrid":
            dense = NaiveRAG(self.vector_store, self.embedder)
            bm25 = BM25RAG(self.bm25_index, self.bm25_chunks)
            candidates = HybridRAG(bm25, dense).retrieve(
                question, top_k=self.config["retrieval"]["final_top_k"] * 2
            )
            reranker = RemoteReranker()
            results = reranker.rerank(question, candidates, top_k=self.config["reranker"]["top_k"])
            return "hybrid", results
        else:
            retriever = NaiveRAG(self.vector_store, self.embedder)
            results = retriever.retrieve(question, top_k=self.config["retrieval"]["dense_top_k"])
            return "naive", results

    def evaluate(self) -> dict:
        """Run full evaluation benchmark across all RAG strategies."""
        logger.info(f"Starting evaluation benchmark for collection '{self.collection_name}'...")
        summary = evaluate(
            self.vector_store,
            self.embedder,
            self.bm25_index,
            self.bm25_chunks,
            self.graph,
            self.config
        )
        print_report(summary)
        return summary

    def unload(self):
        """Unload indexes from RAM memory immediately (Disk storage remains safe)."""
        logger.info(f"Unloading RAM memory for collection '{self.collection_name}'...")
        self.vector_store.index = None
        self.bm25_index = None
        self.bm25_chunks = []
        self.graph = None
        gc.collect()
        logger.info(f"RAM memory freed for collection '{self.collection_name}'.")

    def close(self):
        """Alias for unload()."""
        self.unload()

    def delete_collection(self):
        """Permanently delete collection from both RAM memory and disk storage."""
        logger.info(f"Deleting collection '{self.collection_name}' from RAM and disk...")
        self.unload()
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
            logger.info(f"Collection '{self.collection_name}' directory deleted from disk.")