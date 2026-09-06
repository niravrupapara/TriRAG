from rank_bm25 import BM25Okapi
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Retriever:

    def __init__(self, chunks):
        self.chunks = chunks
        self.text = [chunk["text"] for chunk in chunks]
        self.tokenized = [text.split() for text in self.text]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query, k=5):
        logger.info(f"BM25 lexical retrieval for query: '{query}' (k={k})")
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)

        ranked = list(zip(self.text, scores))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in ranked[:k]]