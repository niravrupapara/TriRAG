from src.engine import TriRAG

if __name__ == "__main__":
    # 1. Initialize TriRAG Engine
    rag = TriRAG(collection_name="my_study_data")

    # 2. Load cached indexes from disk (or run ingestion if missing)
    rag.load_or_ingest("data/raw/sample_test.txt")

    # 3. Ask a question (Smart Router automatically picks the best strategy)
    question = "How does backpropagation relate to gradient descent?"
    strategy_used, results = rag.query(question)

    # 4. Print results
    print(f"\n{'='*60}")
    print(f"Query   : {question}")
    print(f"Strategy: {strategy_used.upper()}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} | Score: {r.get('score', 0.0):.4f} ---")
        print(r["chunk"])
    print(f"{'='*60}\n")
