from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Embedder(BaseEmbedder):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks):
        texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        logger.info(f"Computing embeddings for {len(texts)} document chunks")
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            normalize_embeddings=True
        )[0].tolist()