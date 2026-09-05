from dataclasses import dataclass
from typing import List

import networkx as nx
from rank_bm25 import BM25Okapi

from src.chunking.chunker import chunk_text
from src.graph.builder import build_graph
from src.graph.extractor import extract_all_graph_documents
from src.ingestion.loader import load_document
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Indexes and state produced by a full document ingestion."""

    chunks: List[str]
    bm25_index: BM25Okapi
    graph: nx.DiGraph


class IngestionPipeline:
    """Load a document and build all retrieval indexes for a collection."""

    def __init__(self, config: dict, embedder, vector_store, bm25_store, graph_store):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.graph_store = graph_store

    def ingest(self, file_path: str) -> IngestionResult:
        """Load, chunk, embed, and persist vector, BM25, and graph indexes."""
        logger.info(f"Starting ingestion pipeline | file: {file_path}")

        text = load_document(file_path)
        chunks = chunk_text(
            text,
            strategy=self.config["chunking"]["strategy"],
            chunk_size=self.config["chunking"]["chunk_size"],
            overlap=self.config["chunking"]["chunk_overlap"],
        )

        vectors = self.embedder.embed(chunks)
        self.vector_store.save(vectors, chunks, self.embedder)

        bm25_index = self.bm25_store.save(
            chunks,
            k1=self.config["bm25"]["k1"],
            b=self.config["bm25"]["b"],
        )

        graph_documents = extract_all_graph_documents(chunks, self.config)
        graph = build_graph(graph_documents, self.embedder)
        self.graph_store.save(graph)

        logger.info(f"Ingestion pipeline complete | chunks: {len(chunks)}")
        return IngestionResult(chunks=chunks, bm25_index=bm25_index, graph=graph)
