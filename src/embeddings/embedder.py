import time 
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

class Embedder:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self, model_name: str):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Convert list of texts to embedding vectors."""
        logger.info(f"Generating embeddings for {len(texts)} texts.")
        start_time = time.time()
        
        vectors = self.model.encode(texts, normalize_embeddings=True)
        elapsed_time = time.time() - start_time
        logger.info(f"Generated embeddings in {elapsed_time:.2f} seconds.")
        return vectors
    
    def embed_one(self, text: str) -> np.ndarray:
        """Convert a single text to an embedding vector."""
        logger.debug(f"Generating embedding for a single text.")
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector
