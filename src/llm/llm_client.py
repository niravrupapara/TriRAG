import os
from groq import Groq
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
