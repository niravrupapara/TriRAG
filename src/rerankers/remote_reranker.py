import requests
from .base import BaseReranker


class RemoteReranker(BaseReranker):

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def rerank(self, query, chunks, top_k):
        response = requests.post(
            f"{self.base_url}/rerank",
            json={
                "question": query,
                "candidates": chunks,
                "top_k": top_k
            }
        )

        response.raise_for_status()
        return response.json()["results"]