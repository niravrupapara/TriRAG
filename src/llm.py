import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class LLM:

    def __init__(self, model="open-mistral-7b"):
        logger.info(f"Initializing ChatMistralAI with model: {model}")
        self.llm = ChatMistralAI(
            model=model,
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

    def generate(self, query, context):
        logger.info("Generating response with LLM...")
        prompt = f"""
You are a helpful RAG assistant.

Answer the question using only the provided context.
If the answer is not present in the context, say that you do not know.

Context:
{context}

Question:
{query}

Answer:
"""
        return self.llm.invoke(prompt).content

    def check_relevance(self, query, context):
        logger.info("Checking context relevance with LLM...")
        prompt = f"""
Determine whether the context contains enough relevant information
to answer the question.

Question:
{query}

Context:
{context}

Return only YES or NO.
"""
        response = self.llm.invoke(prompt).content.strip().upper()
        return response.startswith("YES")

    def rewrite_query(self, query):
        logger.info(f"Requesting query rewrite from LLM for: '{query}'")
        prompt = f"""
Rewrite the following question to improve information retrieval.

Keep the original meaning.
Return only the rewritten question.

Question:
{query}
"""
        return self.llm.invoke(prompt).content.strip()