from dotenv import load_dotenv

load_dotenv()

from src.engine import Engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    pdf_path = "data/sample_small.pdf"

    logger.info(f"Starting TriRAG with document: {pdf_path}")
    engine = Engine(pdf_path)
    
    # question = "Why is cross-encoder reranking performed after initial retrieval?"
    question = "What is the capital of Japan?"
    logger.info(f"Running query: {question}")

    answer, chunks = engine.query(question)

    logger.info("Query completed successfully.")
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()