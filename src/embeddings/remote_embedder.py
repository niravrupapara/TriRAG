import requests
import numpy as np
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "http://127.0.0.1:8000"  # Update this to your FastAPI server URL

class RemoteEmbedder:
    """Fetch embeddings from a remote FastAPI server."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        logger.info(f"RemoteEmbedder initialized with base URL: {self.base_url}")   

    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts via the API."""
        logger.info(f"Requesting embeddings for {len(texts)} texts from remote server.")
        response = requests.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts}
        )

        response.raise_for_status()
        vectors = response.json()["vectors"]
        logger.info(f"Received embeddings for {len(vectors)} texts.")
        return np.array(vectors)
    
    def embed_one(self, text: str) -> np.ndarray:
        """Get embedding for a single text via the API."""
        logger.info("Requesting embedding for a single text from remote server.")
        
        return self.embed([text])[0]  # Reuse the batch method for a single text