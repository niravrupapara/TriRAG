import requests
from typing import List, Union, Any

from langchain_core.embeddings import Embeddings
from src.utils.logging import get_logger


logger = get_logger(__name__)


class RemoteEmbedder(Embeddings):

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

        logger.info(
            "RemoteEmbedder initialized: %s",
            self.base_url
        )

    def embed_documents(self, chunks: List[Union[str, dict]]) -> List[List[float]]:
        texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
        logger.info(
            "Requesting embeddings for %d documents",
            len(texts)
        )

        response = requests.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts},
            timeout=120
        )

        response.raise_for_status()

        vectors = response.json()["vectors"]

        logger.info(
            "Received %d document embeddings",
            len(vectors)
        )

        return vectors

    def embed_query(self, text: str) -> List[float]:
        logger.info("Requesting embedding for query")

        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text},
            timeout=60
        )

        response.raise_for_status()

        vector = response.json()["vector"]

        logger.info("Received query embedding")

        return vector
