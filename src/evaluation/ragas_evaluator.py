from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy
)

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


class RagasEvaluator:

    def __init__(self):

        # Ollama evaluator
        self.llm = LangchainLLMWrapper(
            ChatOllama(
                model="phi3:3.8b",
                temperature=0
            )
        )

        # Set LLM directly on legacy metrics
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm

        # Embeddings
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def evaluate_all(self, question, answer, top_chunks):

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [
                [chunk["text"] for chunk in top_chunks]
            ]
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ],
            embeddings=self.embeddings
        )


        return result


if __name__ == "__main__":
    evaluator = RagasEvaluator()

    question = "What is Retrieval-Augmented Generation?"

    answer = (
        "Retrieval-Augmented Generation retrieves relevant documents "
        "and uses them as context for generating an answer."
    )

    top_chunks = [
        {
            "text": (
                "Retrieval-Augmented Generation (RAG) combines "
                "information retrieval with text generation."
            )
    },
    {
        "text": (
            "A retriever searches a knowledge base and returns "
            "documents relevant to the user's question."
        )
    },
    {
        "text": (
            "The retrieved documents are provided to a language "
            "model as context so it can generate a grounded answer."
        )
    },
]
    result = evaluator.evaluate_all(
        question,
        answer,
        top_chunks
    )

    print("\n========== RAGAS RESULT ==========")
    print(result)
    print("===================================")