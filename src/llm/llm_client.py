import os

from typing import List

from langchain_mistralai import ChatMistralAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from dotenv import load_dotenv
from src.utils.logging import get_logger

logger = get_logger(__name__)

load_dotenv()

_api_key = os.getenv("MISTRAL_API_KEY")
if not _api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file.")


_client = None

def _get_client(config: dict) -> ChatMistralAI:
    """Lazily build and cache the shared ChatMistralAI client.""" 

    global _client
    
    if _client is None:
        logger.info(f"Loading ChatMistral client | model: {config['llm']['model']}")
        _client = ChatMistralAI(
            model = config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
            api_key=_api_key
        )
    return _client

def call_llm(prompt: str, config: dict) -> str:
    """send a prompt to Mistral LLM and return the response text."""

    logger.debug(f"Calling LLM | model: {config['llm']['model']} | prompt length: {len(prompt)}")

    client = _get_client(config)

    response = client.invoke(prompt)


    result = response.content.strip()
    logger.debug(f"LLM response received | length: {len(result)}")
    return result


_graph_transformer = None

def call_graph_llm(documents: List[Document], config: dict) -> List[GraphDocument]:
    """Send documents to the graph-extraction LLM and return extracted graph documents."""

    global _graph_transformer

    if _graph_transformer is None:
        logger.info(f"Loading Mistral client for graph extraction | model: {config['llm']['model']}")
        graph_llm = _get_client(config)
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