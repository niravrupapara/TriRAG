from src.utils.logging import get_logger

logger = get_logger(__name__)


class NaiveRetriever:

    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query, k=3):
        logger.info(f"Naive vector retrieval for query: '{query}' (k={k})")
        query_emb = self.embedder.embed_query(query)
        return self.vectorstore.search([query_emb], k)