import requests
from .base import BaseEmbedder


class RemoteEmbedder(BaseEmbedder):

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def embed_documents(self, texts):
        response = requests.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()["vectors"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]