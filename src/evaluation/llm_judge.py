from src.llm.llm_client import call_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT = """You are an expert evaluator for a RAG system.
Given a QUESTION and CONTEXT retrieved by the system, rate the quality of the context.

score from 1 to 5:
1 - Context is completely irrelevant
2 - Context is mostly irrelevant
3 - Context is partially relevant
4 - Context is mostly relevant
5 - Context is perfectly relevant and faithful

QUESTION: {question}

CONTEXT:
{context}

Return ONLY a single integer between 1 and 5. No explanation.
"""

def judge(question: str, results: list, config: dict) -> float:
    """Score retrieved chunks using LLM as judge. Returns score 1.0-5.0."""
    logger.info(f"LLM judge scoring | question: {question[:60]}")

    context ="\n\n".join([r["chunk"] for r in results])
    prompt = JUDGE_PROMPT.format(question=question, context=context)
    
    response = call_llm(prompt, config)

    try:
        score  = float(response.strip())
        if score < 1.0 or score > 5.0:
            raise ValueError(f"Score out of range: {score}")
        logger.info(f"LLM judge score: {score} | question: {question[:60]}")
        return score
    except ValueError as e:
        logger.error(f"Invalid score from LLM judge: {response} | error: {e}")
        return 0.0
    