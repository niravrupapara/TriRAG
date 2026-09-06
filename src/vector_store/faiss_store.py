import faiss
import numpy as np

from src.vector_store.base import BaseVectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaissVectorStore(BaseVectorStore):

    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self, chunks, embeddings):
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Indexed {len(chunks)} vectors into FAISS (total in index: {self.index.ntotal})")

    def search(self, query_embedding, k=5):
        logger.info(f"FAISS searching for top-{k} similar vectors")
        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )

        return [ self.chunks[idx] for idx in indices[0] if idx < len(self.chunks) ]