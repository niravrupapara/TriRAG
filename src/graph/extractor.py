import json
from typing import List, Tuple
from src.llm.llm_client import call_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRIPLE_PROMPT = """Extract factual triples from the TEXT below.
A triple is: [subject, predicate, object]
Rules:
- Extract ONLY from the TEXT provided below, not from examples
- Return ONLY a valid JSON array, no explanation, no extra text
- If no triples found, return empty array: []

Format: [["subject1", "predicate1", "object1"], ["subject2", "predicate2", "object2"]]

TEXT:
{text}

JSON:"""

def extract_triples(chunk: str, config: dict) -> list[Tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from a text chunk using LLM."""
    logger.debug(f"Extracting triples from chunk of length: {len(chunk)}")

    prompt = TRIPLE_PROMPT.format(text=chunk)
    response = call_llm(prompt, config)

    try:
        raw = json.loads(response)
        triples = [(t[0].strip(), t[1].strip(), t[2].strip()) for t in raw if len(t) == 3]
        logger.debug(f"Extracted {len(triples)} triples from chunk.")
        return triples
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Failed to parse triples from LLM response. Skipping chunk")
        return []
    
def extract_all_triples(chunks: List[str], config: dict) -> List[Tuple[str, str, str]]:
    """Extract triples from all chunks and return combined list."""
    logger.info(f"Extracting triples from {len(chunks)} chunks...")

    all_triples = []
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk, config)
        all_triples.extend(triples)
        logger.debug(f"Chunk {i+1}/{len(chunks)} | triples so far: {len(all_triples)}")

    logger.info(f"Triple extraction complete | total triples: {len(all_triples)}")
    return all_triples