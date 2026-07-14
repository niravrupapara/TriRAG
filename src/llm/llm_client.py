import os

from typing import List
from groq import Groq
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from dotenv import load_dotenv
from src.utils.logging import get_logger

logger = get_logger(__name__)

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
_client = Groq(api_key=_api_key)

def call_llm(prompt: str, config: dict) -> str:
    """send a prompt to Groq LLM and return the response text."""

    logger.debug(f"Calling LLM | model: {config['llm']['model']} | prompt length: {len(prompt)}")

    response = _client.chat.completions.create(
        model=config['llm']['model'],
        messages=[{"role": "user", "content": prompt}],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens']
    )

    result = response.choices[0].message.content.strip()
    logger.debug(f"LLM response received | length: {len(result)}")
    return result


_graph_transformer = None

def call_graph_llm(documents: List[Document], config: dict) -> List[GraphDocument]:
    """Send documents to the graph-extraction LLM and return extracted graph documents."""

    global _graph_transformer

    if _graph_transformer is None:
        logger.info(f"Loading ChatGroq client for graph extraction | model: {config['llm']['model']}")
        graph_llm = ChatGroq(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            api_key=_api_key
        )
        _graph_transformer = LLMGraphTransformer(llm=graph_llm)
        logger.info("Graph transformer ready.")

    logger.debug(f"Calling graph LLM | documents: {len(documents)}")
    graph_documents = _graph_transformer.convert_to_graph_documents(documents)
    logger.debug(f"Graph LLM response received | graph_documents: {len(graph_documents)}")
    return graph_documents


def extract_entities(query: str, config: dict) -> List[str]:
    """Extract key entity names from a query using the LLM."""

    prompt = (
        "Extract the key entity or concept names mentioned in this"
        "question. Return them as a comm-separated list with no "
        "extra text.\n\n"
        f"Question: {query}"
    )

    response = call_llm(prompt, config)

    entities = [e.strip() for e in response.split(',') if e.strip()]

    logger.debug(f"Extracted entities: {entities}")

    return entities