
import requests
from typing import List, Union, Any

from src.utils.logging import get_logger


logger = get_logger(__name__)


class RemoteReranker:

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

        logger.info(
            "RemoteReranker initialized: %s",
            self.base_url
        )

    def rerank(
        self,
        query: str,
        candidates: List[Union[str, dict]],
        top_k: int = 3
    ) -> list:

        logger.info(
            "Requesting reranking for %d candidates",
            len(candidates)
        )

        response = requests.post(
            f"{self.base_url}/rerank",
            json={
                "question": query,
                "candidates": candidates,
                "top_k": top_k
            },
            timeout=120
        )

        response.raise_for_status()

        results = response.json()["results"]

        logger.info(
            "Received %d reranked results",
            len(results)
        )

        return results

