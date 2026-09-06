from src.llm import LLM
from src.fusion.rrf import RRF
from src.rerankers.reranker import Reranker
from src.agent.corrective_rag import CorrectiveRAG

from src.loaders.pdf_loader import PDFLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.embedder import Embedder

from src.vector_store.faiss_store import FaissVectorStore

from src.graph.extractor import extract_graph_documents
from src.graph.builder import build_graph

from src.retriever.naive_retriever import NaiveRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.graph_retriever import GraphRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Engine:

    def __init__(self, pdf_path):
        logger.info(f"Initializing TriRAG Engine with document: {pdf_path}")
        self.llm = LLM()
        self.embedder = Embedder()

        text = PDFLoader().load(pdf_path)
        chunks = RecursiveChunker().chunk([text])
        logger.info(f"Loaded and chunked document into {len(chunks)} chunks")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        vectorstore = FaissVectorStore(len(embeddings[0]))
        vectorstore.add(texts, embeddings)
        logger.info("FAISS vector store populated")

        graph_documents = extract_graph_documents(
            chunks,
            self.llm.llm
        )
        graph = build_graph(graph_documents)
        logger.info("Knowledge graph constructed")

        self.naive = NaiveRetriever(
            self.embedder,
            vectorstore
        )
        self.bm25 = BM25Retriever(chunks)
        self.graph = GraphRetriever(
            graph,
            chunks,
            self.embedder
        )

        self.rrf = RRF()
        self.reranker = Reranker()

        self.rag = CorrectiveRAG(
            naive=self.naive,
            bm25=self.bm25,
            graph=self.graph,
            rrf=self.rrf,
            reranker=self.reranker,
            llm=self.llm,
            max_retries=3
        )
        logger.info("Engine initialization complete and ready for queries")

    def query(self, question):
        logger.info(f"Engine processing query: {question}")
        return self.rag.run(question)