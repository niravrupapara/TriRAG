
from typing import List
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

from src.llm.llm_client import call_graph_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_all_graph_documents(chunks: List[str], config: dict) -> List[GraphDocument]:
    """Extract structured graph documents (typed nodes + relationships) from chunks."""
    logger.info(f"Extracting graph documents from {len(chunks)} chunks...")

    documents = [Document(page_content=chunk) for chunk in chunks]
    graph_documents = call_graph_llm(documents, config)

    total_nodes = sum(len(gd.nodes) for gd in graph_documents)
    total_rels = sum(len(gd.relationships) for gd in graph_documents)

    logger.info(f"Graph extraction complete | nodes: {total_nodes} | relationships: {total_rels}")

    return graph_documents
