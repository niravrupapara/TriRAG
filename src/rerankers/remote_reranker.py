import requests
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "http://127.0.0.1:8000"


class RemoteReranker:
    """Fetches reranked results from the running FastAPI service instead of loading the CrossEncoder model locally."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        logger.info(f"RemoteReranker will use API at: {self.base_url}")

    def rerank(self, question: str, candidates: List[dict], top_k: int) -> List[dict]:
        """Rerank candidate chunks via the API."""
        logger.info(f"Requesting rerank for {len(candidates)} candidates via API.")

        response = requests.post(
            f"{self.base_url}/rerank",
            json={
                "question": question,
                "candidates": candidates,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        logger.info(f"Received {len(results)} reranked results from API.")
        return results