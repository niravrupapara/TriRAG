from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Stores and loads a FAISS index from text chunks and their embeddings."""

    def __init__(self, store_path: str):
        """Initialized the vector store with storage directory."""
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.chunks = []
        logger.debug(f"VectorStore initialized at: {self.store_path}")

    def save(self, vectors, chunks: List[str], embedder):
        """Build a FAISS index from precomputed vectors and save it to disk."""
        self.chunks = chunks
        text_embedding_pairs = list(zip(chunks, vectors.tolist()))

        self.index = FAISS.from_embeddings(
            text_embedding_pairs,
            embedder,
            distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        )

        self.index.save_local(str(self.store_path))
        logger.info(f"Saved {len(chunks)} chunks to FAISS index at {self.store_path}")


    def load(self, embedder):
        """Load the FAISS index from disk."""
        self.index = FAISS.load_local(
            str(self.store_path),
            embedder,
            allow_dangerous_deserialization=True
        )
        self.chunks = [doc.page_content for doc in self.index.docstore._dict.values()]
        logger.info(f"Loaded {len(self.chunks)} chunks from FAISS index at {self.store_path}")

    def is_empty(self) -> bool:
        """Check if vector store has a saved FAISS index."""
        return not (self.store_path / "index.faiss").exists()