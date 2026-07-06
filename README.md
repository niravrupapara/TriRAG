# TriRAG — Multi-Strategy RAG Knowledge Engine

A production-inspired RAG system that implements **three retrieval strategies** (Naive, Hybrid, GraphRAG), an **intelligent query router**, and a **benchmarking evaluation harness** — all built from scratch with clean MLOps structure and structured logging.

---

## What is TriRAG?

Most RAG tutorials implement only one retrieval strategy. TriRAG implements **three**, benchmarks them head-to-head, and automatically routes each query to the best strategy.

```
Document → Ingest → [Vector Store | BM25 Index | Knowledge Graph]
                              ↓
Query → Router → Best Strategy → Retrieved Chunks → Answer
                              ↓
              Evaluator → Hit Rate | MRR | NDCG | LLM Judge Score
```

---

## Architecture

```
TriRAG/
├── src/
│   ├── ingestion/          # PDF & TXT document loader
│   ├── chunking/           # Fixed, recursive, semantic chunkers
│   ├── embeddings/         # Sentence-transformer embedder
│   ├── storage/            # Vector store, BM25 store, Graph store
│   ├── retrieval/          # Naive RAG, BM25 RAG, Hybrid RAG, GraphRAG
│   ├── rerankers/          # Cross-encoder reranker
│   ├── graph/              # Triple extractor, builder, traversal
│   ├── llm/                # Groq LLM client
│   ├── routing/            # Query classifier and router
│   └── evaluation/         # Metrics, LLM judge, evaluator
├── data/
│   ├── vector/             # FAISS vector index
│   ├── bm25/               # BM25 pickle index
│   ├── graph/              # NetworkX graph pickle
│   └── eval/               # Evaluation questions dataset
├── config.yaml             # Single source of truth for all settings
├── main.py                 # Entry point
└── requirements.txt
```

---

## Four Phases

### Phase 1 — Naive RAG
Dense vector retrieval using cosine similarity.

```
PDF → chunks → embeddings (BGE-small) → FAISS index → top-k by cosine similarity
```

- Embedding model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Chunking: recursive text splitter (512 tokens, 50 overlap)
- Storage: FAISS flat index saved to disk

---

### Phase 2 — Hybrid RAG
Combines keyword (BM25) and dense retrieval, re-ranked by a cross-encoder.

```
Query → BM25 retrieval + Dense retrieval → RRF fusion → Cross-encoder reranking → top-k
```

- BM25: `rank-bm25` with tunable k1/b parameters
- Fusion: Reciprocal Rank Fusion (RRF)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

### Phase 3 — GraphRAG
Builds a knowledge graph from LLM-extracted triples and retrieves via graph traversal.

```
Chunks → LLM triple extraction → NetworkX DiGraph → ego-graph traversal → results
```

- Triple format: `(subject, predicate, object)`
- LLM: Groq `llama-3.1-8b-instant`
- Traversal: `nx.ego_graph()` with configurable `max_hops`
- Storage: pickle-serialized NetworkX DiGraph

---

### Phase 4 — Router + Evaluation Harness

**Query Router** — classifies each query and routes to the best strategy:

| Query Type | Example | Strategy |
|-----------|---------|----------|
| Relationship query | "how does PCA relate to dimensionality reduction" | GraphRAG |
| Short keyword query | "PCA" | Hybrid RAG |
| General factual query | "explain backpropagation algorithm" | Naive RAG |

**Evaluation Harness** — benchmarks all 3 strategies:

| Metric | What it measures |
|--------|-----------------|
| Hit Rate | Did the correct chunk appear in top-k results? |
| MRR | How high did the correct chunk rank? |
| NDCG | Quality of the full ranking order |
| LLM Score | Answer relevance rated 1-5 by LLM judge |

---

## Benchmark Results

Evaluated on 5 questions from a neural networks PDF:

```
=================================================================
                        EVALUATION REPORT
=================================================================
Strategy       Hit Rate        MRR       NDCG    LLM Score
-----------------------------------------------------------------
NAIVE           1.0000    1.0000    1.0000      4.4000
HYBRID          1.0000    1.0000    1.0000      4.4000
GRAPH           0.2000    0.2000    0.2000      4.0000
=================================================================
```

**Key insight:** Naive and Hybrid RAG excel at direct factual questions. GraphRAG is designed for relationship queries — it performs best when questions ask how entities connect, not what they are.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers` — BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS (via numpy) |
| Keyword Search | `rank-bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Knowledge Graph | `networkx` DiGraph |
| LLM | Groq API — llama-3.1-8b-instant |
| Document Loading | `pypdf` |
| Config | `pyyaml` |
| Env Management | `python-dotenv` |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/niravrupapara/TriRAG.git
cd TriRAG
```

**2. Create and activate conda environment**
```bash
conda create -n rag python=3.11
conda activate rag
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at: https://console.groq.com

**5. Run**
```bash
python main.py
```

---

## Configuration

All settings are in `config.yaml` — no hardcoded values anywhere in the code:

```yaml
chunking:
  strategy: recursive      # fixed | recursive | semantic
  chunk_size: 512
  chunk_overlap: 50

embeddings:
  model: BAAI/bge-small-en-v1.5

llm:
  provider: groq
  model: llama-3.1-8b-instant
  temperature: 0.0

graph:
  top_k: 5
  max_hops: 2
```

---

## Project Highlights

- **3 RAG strategies** implemented from scratch — no LangChain, no LlamaIndex
- **Intelligent router** with zero-latency keyword classification
- **Evaluation harness** with 4 metrics including LLM-as-judge
- **Structured logging** throughout — every step is traceable
- **MLOps structure** — config-driven, modular, each component independently replaceable
- **35+ commits** — built incrementally with clean git history

---

## Author

**Nirav Rupapara**  
[GitHub](https://github.com/niravrupapara)
