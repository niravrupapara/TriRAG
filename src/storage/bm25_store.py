import os
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BM25Store:
    """Save and load BM25 index to disk."""

    def __init__(self, path: str):
        self.path = path
        self.index_path = os.path.join(path, "bm25_index.pkl")
        logger.debug("BM25Store initialized.")

    def save(self, chunks: List[str], k1: float, b: float) -> BM25Okapi:
        """Tokenize chunks, build BM25 index and save to disk."""
        os.makedirs(self.path, exist_ok=True)

        tokenized = [chunk.lower().split() for chunk in chunks]
        index = BM25Okapi(tokenized, k1=k1, b=b)

        with open(self.index_path, "wb") as f:
            pickle.dump({"index": index, "chunks": chunks}, f)

        logger.info(f"BM25 index saved | chunks indexed: {len(chunks)}")

        return index

    def load(self) -> Tuple[BM25Okapi, List[str]]:
        """Load BM25 index and chunks from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"BM25 index not found at: {self.index_path}")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"BM25 index loaded | chunks: {len(data['chunks'])}")
        return data["index"], data["chunks"]
    