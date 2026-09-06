from src.utils.logging import get_logger

logger = get_logger(__name__)


class CorrectiveRAG:

    def __init__(self, naive, bm25, graph, rrf, reranker, llm, max_retries=3):
        self.naive = naive
        self.bm25 = bm25
        self.graph = graph
        self.rrf = rrf
        self.reranker = reranker
        self.llm = llm
        self.max_retries = max_retries

    def retrieve(self, query):
        logger.info(f"Running multi-strategy retrieval for query: '{query}'")
        naive_chunks = self.naive.retrieve(query)
        bm25_chunks = self.bm25.retrieve(query)
        graph_chunks = self.graph.retrieve(query)

        fused_chunks = self.rrf.fuse(
            naive_chunks,
            bm25_chunks,
            graph_chunks
        )

        return self.reranker.rerank(query, fused_chunks)

    def run(self, query):
        current_query = query

        for attempt in range(self.max_retries + 1):
            logger.info(f"Corrective RAG loop [{attempt + 1}/{self.max_retries + 1}] using query: '{current_query}'")
            chunks = self.retrieve(current_query)
            context = "\n\n".join(chunks)

            if self.llm.check_relevance(current_query, context):
                logger.info("Context evaluated as RELEVANT. Generating answer.")
                return self.llm.generate(current_query, context), chunks

            logger.warning("Context evaluated as NOT relevant.")
            if attempt < self.max_retries:
                current_query = self.llm.rewrite_query(current_query)
                logger.info(f"Rewritten query: '{current_query}'")

        logger.warning("Max retries reached. Generating best-effort answer.")
        return self.llm.generate(current_query, context), chunks