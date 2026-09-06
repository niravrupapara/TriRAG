
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_graph_documents(chunks, llm, batch_size=5):
    logger.info(f"Extracting graph relationships from {len(chunks)} chunks (batch_size={batch_size})")
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = []

    documents = [
        Document(page_content=chunk["text"])
        for chunk in chunks
    ]

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        graph_documents.extend(
            transformer.convert_to_graph_documents(batch)
        )

    logger.info(f"Extracted {len(graph_documents)} graph document representations")
    return graph_documents