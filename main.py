from src.engine import TriRAG

if __name__ == "__main__":
    # 1. Initialize TriRAG Engine
    rag = TriRAG(collection_name="my_study_data")

    # 2. Load cached indexes from disk (or run ingestion if missing)
    rag.load_or_ingest("data/raw/sample_test.txt")

    question = "How does backpropagation work in neural networks?"
    strategy_used, answer, results = rag.query_and_generate(question, strategy="hybrid")

    # 4. Print results & LLM Answer
    print(f"\n{'='*60}")
    print(f"Query    : {question}")
    print(f"Strategy : {strategy_used.upper()}")
    print(f"{'='*60}")
    print(f"AI ANSWER:\n{answer}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Context Chunk {i+1} | Score: {r.get('score', 0.0):.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")
