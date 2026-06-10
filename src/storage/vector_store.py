import json 
import numpy as np
from pathlib import Path
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

class VectorStore:
    """Stores and loads text chunks and their embedding vectors."""

    def __init__(self, store_path: str):
        """Initialize the vector store with storage directory."""
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.vectors = None
        self.chunks = []
        logger.debug(f"VectorStore initialized at: {self.store_path}")

    def save(self, vectors:np.ndarray, chunks: List[str]):
        """Save vectors and chunks to disk."""
        self.vectors = vectors
        self.chunks = chunks
        np.save(self.store_path / "vectors.npy", vectors)
        with open(self.store_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks and their vectors to {self.store_path}")

    def load(self):
        """Load vectors and chunks from disk."""
        self.vectors = np.load(self.store_path / "vectors.npy")
        with open(self.store_path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks and their vectors from {self.store_path}")

    def is_empty(self) -> bool:
        """Check if vector store has saved data."""
        return not (self.store_path / "vectors.npy").exists()